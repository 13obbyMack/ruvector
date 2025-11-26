//! MCP request handlers

use super::protocol::*;
use crate::config::Config;
use anyhow::{Context, Result};
use ruvector_core::{
    types::{DbOptions, DistanceMetric, SearchQuery, VectorEntry},
    VectorDB,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// MCP handler state
pub struct McpHandler {
    config: Config,
    databases: Arc<RwLock<HashMap<String, Arc<VectorDB>>>>,
}

impl McpHandler {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            databases: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Handle MCP request
    pub async fn handle_request(&self, request: McpRequest) -> McpResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request.id).await,
            "tools/list" => self.handle_tools_list(request.id).await,
            "tools/call" => self.handle_tools_call(request.id, request.params).await,
            "resources/list" => self.handle_resources_list(request.id).await,
            "resources/read" => self.handle_resources_read(request.id, request.params).await,
            "prompts/list" => self.handle_prompts_list(request.id).await,
            "prompts/get" => self.handle_prompts_get(request.id, request.params).await,
            _ => McpResponse::error(
                request.id,
                McpError::new(error_codes::METHOD_NOT_FOUND, "Method not found"),
            ),
        }
    }

    async fn handle_initialize(&self, id: Option<Value>) -> McpResponse {
        McpResponse::success(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "serverInfo": {
                    "name": "ruvector-mcp",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        )
    }

    async fn handle_tools_list(&self, id: Option<Value>) -> McpResponse {
        let tools = vec![
            McpTool {
                name: "vector_db_create".to_string(),
                description: "Create a new vector database".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Database file path"},
                        "dimensions": {"type": "integer", "description": "Vector dimensions"},
                        "distance_metric": {"type": "string", "enum": ["euclidean", "cosine", "dotproduct", "manhattan"]}
                    },
                    "required": ["path", "dimensions"]
                }),
            },
            McpTool {
                name: "vector_db_insert".to_string(),
                description: "Insert vectors into database".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string"},
                        "vectors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "vector": {"type": "array", "items": {"type": "number"}},
                                    "metadata": {"type": "object"}
                                }
                            }
                        }
                    },
                    "required": ["db_path", "vectors"]
                }),
            },
            McpTool {
                name: "vector_db_search".to_string(),
                description: "Search for similar vectors".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string"},
                        "query": {"type": "array", "items": {"type": "number"}},
                        "k": {"type": "integer", "default": 10},
                        "filter": {"type": "object"}
                    },
                    "required": ["db_path", "query"]
                }),
            },
            McpTool {
                name: "vector_db_stats".to_string(),
                description: "Get database statistics".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string"}
                    },
                    "required": ["db_path"]
                }),
            },
            McpTool {
                name: "vector_db_backup".to_string(),
                description: "Backup database to file".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "db_path": {"type": "string"},
                        "backup_path": {"type": "string"}
                    },
                    "required": ["db_path", "backup_path"]
                }),
            },
        ];

        McpResponse::success(id, json!({ "tools": tools }))
    }

    async fn handle_tools_call(&self, id: Option<Value>, params: Option<Value>) -> McpResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return McpResponse::error(
                    id,
                    McpError::new(error_codes::INVALID_PARAMS, "Missing params"),
                )
            }
        };

        let tool_name = params["name"].as_str().unwrap_or("");
        let arguments = &params["arguments"];

        let result = match tool_name {
            "vector_db_create" => self.tool_create_db(arguments).await,
            "vector_db_insert" => self.tool_insert(arguments).await,
            "vector_db_search" => self.tool_search(arguments).await,
            "vector_db_stats" => self.tool_stats(arguments).await,
            "vector_db_backup" => self.tool_backup(arguments).await,
            _ => Err(anyhow::anyhow!("Unknown tool: {}", tool_name)),
        };

        match result {
            Ok(value) => {
                McpResponse::success(id, json!({ "content": [{"type": "text", "text": value}] }))
            }
            Err(e) => McpResponse::error(
                id,
                McpError::new(error_codes::INTERNAL_ERROR, e.to_string()),
            ),
        }
    }

    async fn handle_resources_list(&self, id: Option<Value>) -> McpResponse {
        McpResponse::success(
            id,
            json!({
                "resources": [
                    {
                        "uri": "database://local/default",
                        "name": "Default Database",
                        "description": "Default vector database",
                        "mimeType": "application/x-ruvector-db"
                    }
                ]
            }),
        )
    }

    async fn handle_resources_read(
        &self,
        id: Option<Value>,
        _params: Option<Value>,
    ) -> McpResponse {
        McpResponse::success(
            id,
            json!({
                "contents": [{
                    "uri": "database://local/default",
                    "mimeType": "application/json",
                    "text": "{\"status\": \"available\"}"
                }]
            }),
        )
    }

    async fn handle_prompts_list(&self, id: Option<Value>) -> McpResponse {
        McpResponse::success(
            id,
            json!({
                "prompts": [
                    {
                        "name": "semantic-search",
                        "description": "Generate a semantic search query",
                        "arguments": [
                            {
                                "name": "query",
                                "description": "Natural language query",
                                "required": true
                            }
                        ]
                    }
                ]
            }),
        )
    }

    async fn handle_prompts_get(&self, id: Option<Value>, _params: Option<Value>) -> McpResponse {
        McpResponse::success(
            id,
            json!({
                "description": "Semantic search template",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": "Search for vectors related to: {{query}}"
                        }
                    }
                ]
            }),
        )
    }

    // Tool implementations
    async fn tool_create_db(&self, args: &Value) -> Result<String> {
        let params: CreateDbParams =
            serde_json::from_value(args.clone()).context("Invalid parameters")?;

        let mut db_options = self.config.to_db_options();
        db_options.storage_path = params.path.clone();
        db_options.dimensions = params.dimensions;

        if let Some(metric) = params.distance_metric {
            db_options.distance_metric = match metric.as_str() {
                "euclidean" => DistanceMetric::Euclidean,
                "cosine" => DistanceMetric::Cosine,
                "dotproduct" => DistanceMetric::DotProduct,
                "manhattan" => DistanceMetric::Manhattan,
                _ => DistanceMetric::Cosine,
            };
        }

        let db = VectorDB::new(db_options)?;
        self.databases
            .write()
            .await
            .insert(params.path.clone(), Arc::new(db));

        Ok(format!("Database created at: {}", params.path))
    }

    async fn tool_insert(&self, args: &Value) -> Result<String> {
        let params: InsertParams = serde_json::from_value(args.clone())?;
        let db = self.get_or_open_db(&params.db_path).await?;

        let entries: Vec<VectorEntry> = params
            .vectors
            .into_iter()
            .map(|v| VectorEntry {
                id: v.id,
                vector: v.vector,
                metadata: v.metadata.and_then(|m| serde_json::from_value(m).ok()),
            })
            .collect();

        let ids = db.insert_batch(entries)?;
        Ok(format!("Inserted {} vectors", ids.len()))
    }

    async fn tool_search(&self, args: &Value) -> Result<String> {
        let params: SearchParams = serde_json::from_value(args.clone())?;
        let db = self.get_or_open_db(&params.db_path).await?;

        let results = db.search(SearchQuery {
            vector: params.query,
            k: params.k,
            filter: params.filter.and_then(|f| serde_json::from_value(f).ok()),
            ef_search: None,
        })?;

        serde_json::to_string_pretty(&results).context("Failed to serialize results")
    }

    async fn tool_stats(&self, args: &Value) -> Result<String> {
        let params: StatsParams = serde_json::from_value(args.clone())?;
        let db = self.get_or_open_db(&params.db_path).await?;

        let count = db.len()?;
        let options = db.options();

        Ok(json!({
            "count": count,
            "dimensions": options.dimensions,
            "distance_metric": format!("{:?}", options.distance_metric),
            "hnsw_enabled": options.hnsw_config.is_some()
        })
        .to_string())
    }

    async fn tool_backup(&self, args: &Value) -> Result<String> {
        let params: BackupParams = serde_json::from_value(args.clone())?;

        std::fs::copy(&params.db_path, &params.backup_path).context("Failed to backup database")?;

        Ok(format!("Backed up to: {}", params.backup_path))
    }

    async fn get_or_open_db(&self, path: &str) -> Result<Arc<VectorDB>> {
        let databases = self.databases.read().await;
        if let Some(db) = databases.get(path) {
            return Ok(db.clone());
        }
        drop(databases);

        // Open new database
        let mut db_options = self.config.to_db_options();
        db_options.storage_path = path.to_string();

        let db = Arc::new(VectorDB::new(db_options)?);
        self.databases
            .write()
            .await
            .insert(path.to_string(), db.clone());

        Ok(db)
    }
}
