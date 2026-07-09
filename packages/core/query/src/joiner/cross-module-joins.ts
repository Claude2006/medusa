import {
  CrossModuleJoinSpec,
  JoinerArgument,
  JoinerRelationship,
  JoinerServiceConfigAlias,
  RemoteJoinerQuery,
} from "@medusajs/types"
import { composeTableName, isObject } from "@medusajs/utils"
import { GraphCatalog } from "./catalog"
import { resolveFieldAliasEntry } from "./helpers"
import {
  InternalJoinerServiceConfig,
  ResidualCrossModuleFilter,
} from "./types"

/**
 * Stage-1 cross-module filtering/sorting (SQL pushdown).
 *
 * Rewrites eligible cross-module filter/sort paths on the joiner query into
 * {@link CrossModuleJoinSpec}s that the root module's DAL turns into
 * correlated EXISTS/scalar subqueries (see
 * `augmentFindOptionsWithCrossModuleJoins`). Pushed-down filters restrict the
 * ROOT rows — matching `query.index` semantics — and are pruned from the
 * query. Expand nodes that existed only to carry those filters are dropped so
 * they never trigger a fetch.
 *
 * A path is eligible when every hop from the root goes through a link module
 * (link table -> target entity pairs), all parties live in the same database,
 * and every filtered/sorted field is `crossjoinable` (a non-computed DML
 * column) on its target entity. Anything else — module-internal hops,
 * read-only links, computed fields, unsupported operators, missing metadata —
 * is reported as residual and left untouched on the query. The in-memory
 * stage (stage 2) will consume `residualCrossModuleFilters` to complete
 * filtering; until then those filters keep today's behavior.
 */

// Operators the DAL cross-module `buildFilterSql` understands.
const SUPPORTED_OPERATORS = new Set([
  "$eq",
  "$ne",
  "$in",
  "$nin",
  "$gt",
  "$gte",
  "$lt",
  "$lte",
  "$like",
  "$ilike",
  "$is",
])

type InternalRelationMetadata = NonNullable<
  NonNullable<JoinerServiceConfigAlias["__internal"]>["relations"]
>[string]

type Hop = {
  kind: "relationship" | "internal"
  property: string
  serviceConfig: InternalJoinerServiceConfig
  entity?: string
  relationship?: JoinerRelationship
  internal?: InternalRelationMetadata
  /**
   * Properties of the contiguous module-internal hops immediately preceding
   * this hop. Used to validate the dotted prefix of read-only link foreign
   * keys (e.g. `items.product_id` requires having traversed `items`).
   */
  internalTrail: string[]
}

type ChainLevel = {
  /** Dotted real (post fieldAlias) path identifying this join level. */
  realPathKey: string
  entity?: string
  spec: CrossModuleJoinSpec
}

type PushdownCandidate = {
  levels: ChainLevel[]
  filters: Record<string, unknown>
}

export function extractCrossModuleJoins(
  {
    query,
    serviceConfig,
  }: {
    query: RemoteJoinerQuery
    serviceConfig: InternalJoinerServiceConfig
  },
  catalog: GraphCatalog
): {
  crossModuleJoins: CrossModuleJoinSpec[]
  residualCrossModuleFilters: ResidualCrossModuleFilter[]
} {
  const registry = new SpecRegistry()
  const residual: ResidualCrossModuleFilter[] = []

  pushDownRootFilters({ query, serviceConfig, catalog, registry, residual })
  pushDownExpandFilters({ query, serviceConfig, catalog, registry, residual })
  pushDownRootOrder({ query, serviceConfig, catalog, registry })

  return {
    crossModuleJoins: registry.specs,
    residualCrossModuleFilters: residual,
  }
}

/**
 * Cross-module filters expressed on the root filters object, e.g.
 * `filters: { price_set: { id: "..." } }` on the `variant` entry point.
 */
function pushDownRootFilters(params: {
  query: RemoteJoinerQuery
  serviceConfig: InternalJoinerServiceConfig
  catalog: GraphCatalog
  registry: SpecRegistry
  residual: ResidualCrossModuleFilter[]
}): void {
  const { query, serviceConfig, catalog, registry, residual } = params

  const filtersArg = getFiltersArg(query.args)
  const rootFilters = filtersArg?.value

  if (!isObject(rootFilters)) {
    return
  }

  for (const [key, value] of Object.entries(rootFilters)) {
    if (key.startsWith("$") || !isObject(value)) {
      continue
    }

    const hops = resolvePathHops(
      { rootConfig: serviceConfig, pathSegments: [key] },
      catalog
    )

    // Same-module paths (and plain object-valued filters) are handled
    // natively by the root module — leave them in place silently.
    if (!hops || !crossesModules(hops, serviceConfig)) {
      continue
    }

    const levels = buildChainLevels({
      hops,
      rootConfig: serviceConfig,
      catalog,
    })

    // Paths that cross modules but cannot be pushed down are residual.
    if (!levels) {
      residual.push({
        path: key,
        filters: value as Record<string, unknown>,
      })
      continue
    }

    const candidates = collectCandidates(
      {
        pathSegments: [key],
        levels,
        filters: value as Record<string, unknown>,
        rootConfig: serviceConfig,
      },
      catalog
    )

    if (!candidates || !registry.registerAll(candidates)) {
      residual.push({ path: key, filters: value as Record<string, unknown> })
      continue
    }

    delete rootFilters[key]
  }
}

/**
 * Cross-module filters that were assigned to nested expand nodes, e.g.
 * `filters: { price_set: { prices: { amount: 10 } } }` lands on the
 * `price_set.prices` expand.
 */
function pushDownExpandFilters(params: {
  query: RemoteJoinerQuery
  serviceConfig: InternalJoinerServiceConfig
  catalog: GraphCatalog
  registry: SpecRegistry
  residual: ResidualCrossModuleFilter[]
}): void {
  const { query, serviceConfig, catalog, registry, residual } = params

  // Iterate a snapshot: pruning removes entries from query.expands.
  const expandsWithFilters = (query.expands ?? []).filter((expand) => {
    const filters = getFiltersArg(expand.args)?.value
    return isObject(filters) && Object.keys(filters).length > 0
  })

  for (const expand of expandsWithFilters) {
    const filters = getFiltersArg(expand.args)!.value as Record<
      string,
      unknown
    >
    const pathSegments = expand.property.split(".")

    const hops = resolvePathHops(
      { rootConfig: serviceConfig, pathSegments },
      catalog
    )

    if (!hops || !crossesModules(hops, serviceConfig)) {
      // Same-module expand filters keep their native behavior (they filter
      // the children fetch within the module).
      continue
    }

    const levels = buildChainLevels({
      hops,
      rootConfig: serviceConfig,
      catalog,
    })

    const candidates =
      levels &&
      collectCandidates(
        { pathSegments, levels, filters, rootConfig: serviceConfig },
        catalog
      )

    if (!candidates || !registry.registerAll(candidates)) {
      residual.push({ path: expand.property, filters })
      continue
    }

    pruneFiltersFromExpand(query, expand)
  }
}

/**
 * Cross-module sorting on the root order arg, e.g.
 * `order: { "price_set.id": "ASC" }` or `order: { price_set: { id: "ASC" } }`.
 * Rewrites pushable entries to the flat `<target table>.<column>` form the
 * DAL's orderBy transform expects.
 */
function pushDownRootOrder(params: {
  query: RemoteJoinerQuery
  serviceConfig: InternalJoinerServiceConfig
  catalog: GraphCatalog
  registry: SpecRegistry
}): void {
  const { query, serviceConfig, catalog, registry } = params

  const orderArg = query.args?.find((arg) => arg.name === "order")
  const order = orderArg?.value

  if (!isObject(order)) {
    return
  }

  const rewrites: Record<string, "ASC" | "DESC"> = {}

  const visit = (obj: Record<string, any>, pathPrefix: string[]): void => {
    for (const key of Object.keys(obj)) {
      const value = obj[key]
      const segments = [...pathPrefix, ...key.split(".")]

      if (isObject(value)) {
        visit(value, segments)
        if (!Object.keys(value).length) {
          delete obj[key]
        }
        continue
      }

      if (
        typeof value !== "string" ||
        !/^(asc|desc)$/i.test(value) ||
        segments.length < 2
      ) {
        continue
      }

      const relationSegments = segments.slice(0, -1)
      const field = segments[segments.length - 1]

      const hops = resolvePathHops(
        { rootConfig: serviceConfig, pathSegments: relationSegments },
        catalog
      )

      // Same-module order paths keep their native behavior.
      if (!hops || !crossesModules(hops, serviceConfig)) {
        continue
      }

      const levels = buildChainLevels({
        hops,
        rootConfig: serviceConfig,
        catalog,
      })

      if (!levels) {
        continue
      }

      const leaf = levels[levels.length - 1]
      const metadata = catalog.getAliasMetadata(leaf.entity)

      if (!metadata?.crossjoinable?.includes(field)) {
        continue
      }

      if (!registry.registerAll([{ levels, filters: {} }])) {
        continue
      }

      delete obj[key]
      rewrites[`${leaf.spec.target.table}.${field}`] = value.toUpperCase() as
        | "ASC"
        | "DESC"
    }
  }

  visit(order, [])
  Object.assign(order, rewrites)
}

/**
 * Splits a filters object located at `pathSegments` into pushdown candidates:
 * field filters apply to the path's own target, while plain-object values
 * whose key resolves to a further link-module hop become deeper candidates.
 *
 * All-or-nothing: returns undefined when anything at this location cannot be
 * pushed down, so a location is never left half-applied.
 */
function collectCandidates(
  params: {
    pathSegments: string[]
    levels: ChainLevel[]
    filters: Record<string, unknown>
    rootConfig: InternalJoinerServiceConfig
  },
  catalog: GraphCatalog
): PushdownCandidate[] | undefined {
  const { pathSegments, levels, filters, rootConfig } = params

  const leaf = levels[levels.length - 1]
  const metadata = catalog.getAliasMetadata(leaf.entity)
  const crossjoinable = new Set(metadata?.crossjoinable ?? [])

  const fieldFilters: Record<string, unknown> = {}
  const candidates: PushdownCandidate[] = []

  for (const [key, value] of Object.entries(filters)) {
    const isRelationCandidate =
      !key.startsWith("$") && !crossjoinable.has(key) && isObject(value)

    if (isRelationCandidate) {
      const deeperSegments = [...pathSegments, key]
      const hops = resolvePathHops(
        { rootConfig, pathSegments: deeperSegments },
        catalog
      )
      const deeperLevels =
        hops && buildChainLevels({ hops, rootConfig, catalog })

      if (!deeperLevels) {
        return undefined
      }

      const nested = collectCandidates(
        {
          pathSegments: deeperSegments,
          levels: deeperLevels,
          filters: value as Record<string, unknown>,
          rootConfig,
        },
        catalog
      )

      if (!nested) {
        return undefined
      }

      candidates.push(...nested)
      continue
    }

    fieldFilters[key] = value
  }

  if (!areFiltersPushable(fieldFilters, crossjoinable)) {
    return undefined
  }

  // Even without own filters the level must be registered so deeper
  // candidates can correlate to it through `parent`.
  candidates.unshift({ levels, filters: fieldFilters })

  return candidates
}

/**
 * Resolves an alias-form relation path into per-hop info, expanding
 * fieldAlias shortcuts (e.g. `price_set` -> `price_set_link.price_set`) the
 * same way compile's parseProperties does. A segment resolves either to a
 * module-internal DML relation (from alias `__internal.relations` metadata)
 * or to a catalog relationship (link modules, read-only links). Returns
 * undefined when a segment resolves to neither, or when the graph schema
 * disagrees with the DML relation metadata.
 */
function resolvePathHops(
  params: {
    rootConfig: InternalJoinerServiceConfig
    pathSegments: string[]
  },
  catalog: GraphCatalog
): Hop[] | undefined {
  const { rootConfig, pathSegments } = params

  const hops: Hop[] = []
  let currentConfig = rootConfig
  let entity = rootConfig.entity
  let internalTrail: string[] = []

  for (const segment of pathSegments) {
    const aliasEntry = resolveFieldAliasEntry(
      currentConfig.fieldAlias?.[segment],
      entity
    )
    const realSegments = aliasEntry ? aliasEntry.path.split(".") : [segment]

    for (const realSegment of realSegments) {
      const internalRelation =
        catalog.getAliasMetadata(entity)?.relations?.[realSegment]

      if (internalRelation) {
        // The graph schema must agree with the DML relation. Modules that
        // remap relations in a custom schema (e.g. order.items resolves to
        // OrderLineItem while the DML relation targets OrderItem) cannot be
        // traversed from the derived metadata.
        const schemaEntity = entity
          ? catalog.getEntity(entity, realSegment)
          : undefined
        if (schemaEntity && schemaEntity !== internalRelation.entity) {
          return undefined
        }

        hops.push({
          kind: "internal",
          property: realSegment,
          serviceConfig: currentConfig,
          entity: internalRelation.entity,
          internal: internalRelation,
          internalTrail: [...internalTrail],
        })

        internalTrail.push(realSegment)
        entity = internalRelation.entity
        continue
      }

      let hopEntity = entity
        ? catalog.getEntity(entity, realSegment) ?? entity
        : entity

      const relationship = catalog.getEntityRelationship({
        parentServiceConfig: currentConfig,
        property: realSegment,
        entity: hopEntity,
      })

      if (!relationship) {
        return undefined
      }

      hopEntity = relationship.entity ?? hopEntity

      const nextConfig = catalog.getServiceConfig({
        serviceName: relationship.serviceName,
        entity: relationship.entity,
      })

      if (!nextConfig) {
        return undefined
      }

      hops.push({
        kind: "relationship",
        property: realSegment,
        relationship,
        serviceConfig: nextConfig,
        entity: hopEntity,
        internalTrail: [...internalTrail],
      })

      internalTrail = []
      currentConfig = nextConfig
      entity = hopEntity
    }
  }

  return hops.length ? hops : undefined
}

/**
 * Maps the resolved hops onto {@link CrossModuleJoinSpec}s, chaining via
 * `parent`. Three hop shapes are supported:
 *
 * - (link module -> target) pairs: the link table joins the current position
 *   to an entity of another module.
 * - Read-only link hops: an FK column on the current position's table points
 *   at another module's entity. When the FK follows an internal hasMany hop,
 *   both fuse into a single spec with the internal child table acting as the
 *   link table (e.g. cart -> items -> product becomes
 *   `cart_line_item(cart_id -> product_id) -> product`).
 * - Module-internal DML relation hops (hasMany/belongsTo), standing alone as
 *   self-join levels so deeper levels can chain on them.
 */
function buildChainLevels(params: {
  hops: Hop[]
  rootConfig: InternalJoinerServiceConfig
  catalog: GraphCatalog
}): ChainLevel[] | undefined {
  const { hops, rootConfig, catalog } = params

  if (!hops.length) {
    return undefined
  }

  const levels: ChainLevel[] = []
  const realPath: string[] = []
  // The DAL correlates the root EXISTS against the entity's "id" column and
  // chained joins against the parent target's primaryKey.
  let correlateKey = "id"
  let parentTable: string | undefined
  let currentServiceName = rootConfig.serviceName

  const pushLevel = (
    spec: CrossModuleJoinSpec,
    hopEntity: string | undefined,
    properties: string[],
    serviceName: string
  ) => {
    realPath.push(...properties)
    levels.push({
      realPathKey: realPath.join("."),
      entity: hopEntity,
      spec: parentTable ? { parent: parentTable, ...spec } : spec,
    })
    parentTable = spec.target.table
    correlateKey = spec.target.primaryKey ?? "id"
    currentServiceName = serviceName
  }

  let i = 0
  while (i < hops.length) {
    const hop = hops[i]

    if (hop.kind === "internal") {
      const relation = hop.internal!
      const tableMetadata = catalog.getAliasMetadata(hop.entity)

      if (!tableMetadata?.tableName || !isSimpleKey(relation.foreignKey)) {
        return undefined
      }

      if (relation.foreignKeyOwner === "target") {
        // hasMany: the child's FK column references the parent's id column.
        if (correlateKey !== "id") {
          return undefined
        }

        const next = hops[i + 1]
        const isFusableNext =
          next?.kind === "relationship" && !next.serviceConfig.isLink

        if (isFusableNext) {
          // Fuse with the following read-only link hop: the child table acts
          // as the link table towards the external module.
          const fkHop = resolveFkHopParts(
            next,
            { rootConfig, fromServiceName: currentServiceName },
            catalog
          )
          if (!fkHop) {
            return undefined
          }

          pushLevel(
            {
              link: {
                table: tableMetadata.tableName,
                sourceKey: relation.foreignKey,
                targetKey: fkHop.column,
              },
              target: fkHop.target,
            },
            next.entity,
            [hop.property, next.property],
            next.serviceConfig.serviceName
          )
          i += 2
          continue
        }

        // Standalone internal hop: a self-join level deeper levels chain on.
        pushLevel(
          {
            link: {
              table: tableMetadata.tableName,
              sourceKey: relation.foreignKey,
              targetKey: "id",
            },
            target: {
              table: tableMetadata.tableName,
              ...(tableMetadata.schema ? { schema: tableMetadata.schema } : {}),
              primaryKey: "id",
            },
          },
          hop.entity,
          [hop.property],
          currentServiceName
        )
        i += 1
        continue
      }

      // belongsTo: the FK column lives on the current position's table.
      const currentTable =
        parentTable ?? catalog.getAliasMetadata(rootConfig.entity)?.tableName
      if (!currentTable) {
        return undefined
      }

      pushLevel(
        {
          link: {
            table: currentTable,
            sourceKey: correlateKey,
            targetKey: relation.foreignKey,
          },
          target: {
            table: tableMetadata.tableName,
            ...(tableMetadata.schema ? { schema: tableMetadata.schema } : {}),
            primaryKey: "id",
          },
        },
        hop.entity,
        [hop.property],
        currentServiceName
      )
      i += 1
      continue
    }

    if (hop.serviceConfig.isLink) {
      const linkConfig = hop.serviceConfig
      const targetHop = hops[i + 1]

      if (!targetHop || targetHop.kind !== "relationship") {
        return undefined
      }

      const linkRel = hop.relationship!
      const targetRel = targetHop.relationship!

      if (linkRel.inverse || targetRel.inverse) {
        return undefined
      }

      if (
        !isSimpleKey(linkRel.primaryKey) ||
        !isSimpleKey(linkRel.foreignKey) ||
        !isSimpleKey(targetRel.primaryKey) ||
        !isSimpleKey(targetRel.foreignKey)
      ) {
        return undefined
      }

      if (linkRel.foreignKey !== correlateKey) {
        return undefined
      }

      // The root and target modules must live in the same database. Link
      // modules are not compared: they always run on the app's shared
      // connection and their resolved databaseClientUrl is not reliable.
      if (
        targetHop.serviceConfig.databaseClientUrl !==
        rootConfig.databaseClientUrl
      ) {
        return undefined
      }

      const metadata = catalog.getAliasMetadata(targetHop.entity)
      if (!metadata?.tableName) {
        return undefined
      }

      const linkTable = resolveLinkTableName(linkConfig)
      if (!linkTable) {
        return undefined
      }

      pushLevel(
        {
          link: {
            table: linkTable,
            sourceKey: linkRel.primaryKey,
            targetKey: targetRel.foreignKey,
          },
          target: {
            table: metadata.tableName,
            ...(metadata.schema ? { schema: metadata.schema } : {}),
            primaryKey: targetRel.primaryKey,
          },
        },
        targetHop.entity,
        [hop.property, targetHop.property],
        targetHop.serviceConfig.serviceName
      )
      i += 2
      continue
    }

    // Read-only link hop directly off the current position's table.
    const fkHop = resolveFkHopParts(
      hop,
      { rootConfig, fromServiceName: currentServiceName },
      catalog
    )
    const currentTable =
      parentTable ?? catalog.getAliasMetadata(rootConfig.entity)?.tableName

    if (!fkHop || !currentTable) {
      return undefined
    }

    pushLevel(
      {
        link: {
          table: currentTable,
          sourceKey: correlateKey,
          targetKey: fkHop.column,
        },
        target: fkHop.target,
      },
      hop.entity,
      [hop.property],
      hop.serviceConfig.serviceName
    )
    i += 1
  }

  return levels
}

/**
 * Validates a read-only link hop — an FK column on the current position's
 * table referencing another module's entity — and resolves its join column
 * and target table.
 */
function resolveFkHopParts(
  hop: Hop,
  context: {
    rootConfig: InternalJoinerServiceConfig
    fromServiceName?: string
  },
  catalog: GraphCatalog
): { column: string; target: CrossModuleJoinSpec["target"] } | undefined {
  const { rootConfig, fromServiceName } = context
  const relationship = hop.relationship!

  if (relationship.inverse) {
    return undefined
  }

  // Same-service relationship hops are only traversable through internal
  // relation metadata; alias-generated relationships carry fabricated keys.
  if (relationship.serviceName === fromServiceName) {
    return undefined
  }

  const fkSegments = relationship.foreignKey.split(".")
  const column = fkSegments[fkSegments.length - 1]
  const prefix = fkSegments.slice(0, -1)

  // The dotted prefix declares the relation path the FK column lives on
  // (e.g. `items.product_id`) — it must match the internal hops traversed to
  // reach this position.
  if (prefix.join(".") !== hop.internalTrail.join(".")) {
    return undefined
  }

  if (column.includes(",") || !isSimpleKey(relationship.primaryKey)) {
    return undefined
  }

  const targetMetadata = catalog.getAliasMetadata(hop.entity)
  if (!targetMetadata?.tableName) {
    return undefined
  }

  // The resolved entity must actually belong to the relationship's target
  // module, otherwise the table would be wrong.
  const entityConfig = hop.entity
    ? catalog.getServiceConfig({ entity: hop.entity })
    : undefined
  if (entityConfig?.serviceName !== relationship.serviceName) {
    return undefined
  }

  if (hop.serviceConfig.databaseClientUrl !== rootConfig.databaseClientUrl) {
    return undefined
  }

  return {
    column,
    target: {
      table: targetMetadata.tableName,
      ...(targetMetadata.schema ? { schema: targetMetadata.schema } : {}),
      primaryKey: relationship.primaryKey,
    },
  }
}

function resolveLinkTableName(
  linkConfig: InternalJoinerServiceConfig
): string | undefined {
  if (linkConfig.databaseConfig?.tableName) {
    return linkConfig.databaseConfig.tableName
  }

  const relationships = linkConfig.relationships
    ? Array.from(linkConfig.relationships.values()).flat()
    : []

  // Exclude the alias-generated self relationships the catalog adds.
  const moduleRelationships = relationships.filter(
    (rel) => rel.serviceName !== linkConfig.serviceName
  )

  if (moduleRelationships.length !== 2) {
    return undefined
  }

  const [primary, foreign] = moduleRelationships

  // Mirrors the default naming in link-modules' generateEntity.
  return composeTableName(
    primary.serviceName,
    primary.foreignKey,
    foreign.serviceName,
    foreign.foreignKey
  ).toLowerCase()
}

function areFiltersPushable(
  filters: Record<string, unknown>,
  crossjoinable: Set<string>
): boolean {
  return Object.entries(filters).every(([key, value]) => {
    if (key === "$and" || key === "$or") {
      return (
        Array.isArray(value) &&
        value.every(
          (condition) =>
            isObject(condition) &&
            areFiltersPushable(
              condition as Record<string, unknown>,
              crossjoinable
            )
        )
      )
    }

    if (key.startsWith("$")) {
      return false
    }

    if (!crossjoinable.has(key)) {
      return false
    }

    return isPushableFilterValue(value)
  })
}

function isPushableFilterValue(value: unknown): boolean {
  if (Array.isArray(value)) {
    return value.every((entry) => !isObject(entry))
  }

  if (isObject(value)) {
    const operators = Object.keys(value)

    if (!operators.length) {
      return false
    }

    return operators.every(
      (operator) =>
        SUPPORTED_OPERATORS.has(operator) &&
        isPushableOperatorValue((value as Record<string, unknown>)[operator])
    )
  }

  return true
}

function isPushableOperatorValue(value: unknown): boolean {
  if (Array.isArray(value)) {
    return value.every((entry) => !isObject(entry))
  }

  return !isObject(value)
}

function isSimpleKey(key: string): boolean {
  return !key.includes(",") && !key.includes(".")
}

function crossesModules(
  hops: Hop[] | undefined,
  rootConfig: InternalJoinerServiceConfig
): boolean {
  return !!hops?.some(
    (hop) => hop.serviceConfig.serviceName !== rootConfig.serviceName
  )
}

function getFiltersArg(
  args?: JoinerArgument[]
): JoinerArgument | undefined {
  return args?.find((arg) => arg.name === "filters")
}

/**
 * Removes the pushed-down filters arg from the expand, then drops the expand
 * (and any ancestors that only existed to reach it) when nothing else —
 * requested fields, other args, or remaining descendants — needs the fetch.
 */
function pruneFiltersFromExpand(
  query: RemoteJoinerQuery,
  expand: NonNullable<RemoteJoinerQuery["expands"]>[number]
): void {
  const remainingArgs = (expand.args ?? []).filter(
    (arg) => arg.name !== "filters"
  )
  if (remainingArgs.length) {
    expand.args = remainingArgs
  } else {
    delete expand.args
  }

  const expands = query.expands!

  let property: string | undefined = expand.property
  while (property) {
    const node = expands.find((entry) => entry.property === property)
    if (!node) {
      break
    }

    const prefix = property + "."
    const hasDescendants = expands.some(
      (entry) => entry !== node && entry.property.startsWith(prefix)
    )

    if (
      hasDescendants ||
      node.fields?.length ||
      node.args?.length ||
      node.directives
    ) {
      break
    }

    expands.splice(expands.indexOf(node), 1)
    property = property.includes(".")
      ? property.slice(0, property.lastIndexOf("."))
      : undefined
  }
}

/**
 * Accumulates specs across all pushdown candidates of a query, deduplicating
 * join levels shared between paths and enforcing the DAL's unique-target-table
 * constraint. Registration is atomic per candidate batch.
 */
class SpecRegistry {
  readonly specs: CrossModuleJoinSpec[] = []
  private specByRealPath = new Map<string, CrossModuleJoinSpec>()
  private usedTargetTables = new Set<string>()

  registerAll(candidates: PushdownCandidate[]): boolean {
    // Dry-run the whole batch before committing anything.
    const pendingTables = new Set(this.usedTargetTables)
    const pendingPaths = new Set(this.specByRealPath.keys())

    for (const candidate of candidates) {
      for (const level of candidate.levels) {
        if (pendingPaths.has(level.realPathKey)) {
          continue
        }

        if (pendingTables.has(level.spec.target.table)) {
          return false
        }

        pendingPaths.add(level.realPathKey)
        pendingTables.add(level.spec.target.table)
      }
    }

    for (const candidate of candidates) {
      let leafSpec: CrossModuleJoinSpec | undefined

      for (const level of candidate.levels) {
        let spec = this.specByRealPath.get(level.realPathKey)

        if (!spec) {
          spec = level.spec
          this.specByRealPath.set(level.realPathKey, spec)
          this.usedTargetTables.add(spec.target.table)
          this.specs.push(spec)
        }

        leafSpec = spec
      }

      if (leafSpec && Object.keys(candidate.filters).length) {
        leafSpec.target.filters = leafSpec.target.filters
          ? { $and: [leafSpec.target.filters, candidate.filters] }
          : candidate.filters
      }
    }

    return true
  }
}
