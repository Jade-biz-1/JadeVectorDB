// src/types/api.d.ts
// TypeScript interfaces for all JadeVectorDB API request / response shapes.
// The project is written in JavaScript (allowJs: true) so these declarations
// serve as authoritative documentation and enable editor type-checking via JSDoc.

// ─── Shared primitives ────────────────────────────────────────────────────────

/** ISO-8601 timestamp string, e.g. "2026-04-13T10:00:00Z" */
export type ISOTimestamp = string;

/** Opaque identifier string */
export type ID = string;

// ─── Databases ───────────────────────────────────────────────────────────────

export interface DatabaseStats {
  vectorCount: number;
  indexCount?: number;
}

export interface Database {
  databaseId: ID;
  name: string;
  description?: string;
  vectorDimension: number;
  indexType?: 'FLAT' | 'HNSW' | 'IVF';
  status?: 'active' | 'inactive' | 'error';
  stats?: DatabaseStats;
  createdAt?: ISOTimestamp;
  updatedAt?: ISOTimestamp;
}

export interface ListDatabasesResponse {
  databases: Database[];
  total?: number;
}

export interface CreateDatabaseRequest {
  name: string;
  description?: string;
  vectorDimension: number;
  indexType?: 'FLAT' | 'HNSW' | 'IVF';
}

// ─── Vectors ─────────────────────────────────────────────────────────────────

export type VectorMetadata = Record<string, unknown>;

export interface Vector {
  id: ID;
  /** Floating-point embedding values */
  values: number[];
  metadata?: VectorMetadata;
  createdAt?: ISOTimestamp;
  updatedAt?: ISOTimestamp;
}

export interface ListVectorsResponse {
  vectors: Vector[];
  total: number;
}

export interface StoreVectorRequest {
  id?: ID;
  values: number[];
  metadata?: VectorMetadata;
}

export interface BatchStoreRequest {
  vectors: StoreVectorRequest[];
  upsert?: boolean;
}

export interface BatchStoreResponse {
  count: number;
  stored?: number;
  failed?: number;
}

// ─── Search ──────────────────────────────────────────────────────────────────

export interface SimilaritySearchRequest {
  queryVector: number[];
  topK?: number;
  threshold?: number;
  includeMetadata?: boolean;
  includeVectorData?: boolean;
  includeValues?: boolean;
}

export interface SearchResult {
  vectorId: ID;
  score: number;
  metadata?: VectorMetadata;
  values?: number[];
}

export interface SimilaritySearchResponse {
  results: SearchResult[];
  total?: number;
}

// ─── Authentication ───────────────────────────────────────────────────────────

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  token: string;
  user_id: ID;
  username: string;
  must_change_password?: boolean;
}

export interface RegisterRequest {
  username: string;
  password: string;
  email?: string;
  roles?: string[];
}

export interface RegisterResponse {
  user_id: ID;
  username: string;
  message?: string;
}

export interface ChangePasswordRequest {
  old_password: string;
  new_password: string;
}

export interface ForgotPasswordRequest {
  username: string;
  email?: string;
}

export interface ForgotPasswordResponse {
  reset_token: string;
  message: string;
}

export interface ResetPasswordRequest {
  user_id: ID;
  reset_token: string;
  new_password: string;
}

// ─── Users ───────────────────────────────────────────────────────────────────

export interface User {
  user_id: ID;
  username: string;
  email?: string;
  roles: string[];
  active: boolean;
  createdAt?: ISOTimestamp;
}

export interface ListUsersResponse {
  users: User[];
  total?: number;
}

export interface CreateUserRequest {
  username: string;
  password: string;
  email?: string;
  roles?: string[];
}

// ─── API Keys ────────────────────────────────────────────────────────────────

export interface ApiKey {
  keyId: ID;
  name: string;
  permissions: string[];
  createdAt: ISOTimestamp;
  lastUsed?: ISOTimestamp;
}

export interface ListApiKeysResponse {
  apiKeys: ApiKey[];
}

export interface CreateApiKeyRequest {
  name: string;
  permissions: string[];
}

export interface CreateApiKeyResponse {
  apiKey: string;   // the raw key value — shown only once
  keyId: ID;
  name: string;
  permissions: string[];
}

// ─── Alerting ────────────────────────────────────────────────────────────────

export type AlertType = 'error' | 'warning' | 'info';

export interface Alert {
  id: ID;
  type: AlertType;
  message: string;
  timestamp: ISOTimestamp;
  acknowledged?: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: ISOTimestamp;
}

export interface ListAlertsResponse {
  alerts: Alert[];
  total?: number;
}

// ─── Security / Audit Logs ───────────────────────────────────────────────────

export type AuditLogStatus = 'success' | 'failure';

export interface AuditLog {
  id: ID;
  timestamp: ISOTimestamp;
  user: string;
  event: string;
  status: AuditLogStatus;
  details?: string;
  ipAddress?: string;
}

export interface ListAuditLogsResponse {
  logs: AuditLog[];
  total?: number;
}

// ─── Monitoring ──────────────────────────────────────────────────────────────

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'down';
  uptime?: number;
  version?: string;
  databaseCount?: number;
}

export interface SystemStats {
  cpuUsage?: number;
  memoryUsage?: number;
  diskUsage?: number;
  totalVectors?: number;
}

// ─── Performance ─────────────────────────────────────────────────────────────

export interface PerformanceMetric {
  label: string;
  value: number;
  unit?: string;
  timestamp?: ISOTimestamp;
}

export interface GetMetricsResponse {
  metrics: PerformanceMetric[] | Record<string, number>;
}

// ─── Cluster ─────────────────────────────────────────────────────────────────

export type NodeStatus = 'online' | 'offline' | 'degraded';

export interface ClusterNode {
  nodeId: ID;
  host: string;
  port: number;
  status: NodeStatus;
  role?: 'primary' | 'replica';
  lastSeen?: ISOTimestamp;
}

export interface ListNodesResponse {
  nodes: ClusterNode[];
}

// ─── useDatabases hook ───────────────────────────────────────────────────────

/** Normalised database entry returned by the useDatabases hook */
export interface DatabaseOption {
  id: ID;
  name: string;
}

export interface UseDatabasesResult {
  databases: DatabaseOption[];
  loading: boolean;
  error: string;
  refetch: () => void;
}
