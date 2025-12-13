#!/usr/bin/env python3
"""
JadeVectorDB Distributed Cluster Management CLI
Provides commands for managing distributed clusters, nodes, shards, and diagnostics
"""

import argparse
import json
import sys
from typing import Any, Dict, List

import requests


class ClusterCLI:
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.base_url = f"http://{host}:{port}/api/v1"
    
    def cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status"""
        try:
            response = requests.get(f"{self.base_url}/cluster/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def list_nodes(self) -> List[Dict[str, Any]]:
        """List all nodes in the cluster"""
        try:
            response = requests.get(f"{self.base_url}/cluster/nodes")
            response.raise_for_status()
            return response.json().get("nodes", [])
        except Exception as e:
            return [{"error": str(e)}]
    
    def add_node(self, node_id: str, address: str) -> Dict[str, Any]:
        """Add a new node to the cluster"""
        try:
            payload = {"node_id": node_id, "address": address}
            response = requests.post(f"{self.base_url}/cluster/nodes", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def remove_node(self, node_id: str) -> Dict[str, Any]:
        """Remove a node from the cluster"""
        try:
            response = requests.delete(f"{self.base_url}/cluster/nodes/{node_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def node_health(self, node_id: str) -> Dict[str, Any]:
        """Get health status of a specific node"""
        try:
            response = requests.get(f"{self.base_url}/cluster/nodes/{node_id}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def list_shards(self, database_id: str = None) -> List[Dict[str, Any]]:
        """List all shards, optionally filtered by database"""
        try:
            url = f"{self.base_url}/cluster/shards"
            if database_id:
                url += f"?database_id={database_id}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json().get("shards", [])
        except Exception as e:
            return [{"error": str(e)}]
    
    def migrate_shard(self, shard_id: str, target_node: str) -> Dict[str, Any]:
        """Migrate a shard to a different node"""
        try:
            payload = {"target_node": target_node}
            response = requests.post(f"{self.base_url}/cluster/shards/{shard_id}/migrate", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def shard_status(self, shard_id: str) -> Dict[str, Any]:
        """Get status of a specific shard"""
        try:
            response = requests.get(f"{self.base_url}/cluster/shards/{shard_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def diagnostics(self) -> Dict[str, Any]:
        """Get cluster diagnostics information"""
        try:
            response = requests.get(f"{self.base_url}/cluster/diagnostics")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def metrics(self) -> Dict[str, Any]:
        """Get cluster metrics"""
        try:
            response = requests.get(f"{self.base_url}/cluster/metrics")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def format_output(data: Any, format: str = "json") -> str:
    """Format output data"""
    if format == "json":
        return json.dumps(data, indent=2)
    elif format == "table":
        # Simple table formatting
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                keys = data[0].keys()
                rows = [keys] + [[str(item.get(k, "")) for k in keys] for item in data]
                widths = [max(len(str(row[i])) for row in rows) for i in range(len(keys))]
                lines = []
                for row in rows:
                    lines.append(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
                    if row == rows[0]:
                        lines.append("-" * len(lines[0]))
                return "\n".join(lines)
        return str(data)
    else:
        return str(data)

def main():
    parser = argparse.ArgumentParser(description="JadeVectorDB Cluster Management CLI")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8080, help="API port")
    parser.add_argument("--format", choices=["json", "table"], default="json", help="Output format")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Cluster commands
    subparsers.add_parser("status", help="Get cluster status")
    subparsers.add_parser("diagnostics", help="Get cluster diagnostics")
    subparsers.add_parser("metrics", help="Get cluster metrics")
    
    # Node commands
    subparsers.add_parser("nodes", help="List all nodes")
    
    add_node_parser = subparsers.add_parser("add-node", help="Add a new node")
    add_node_parser.add_argument("node_id", help="Node ID")
    add_node_parser.add_argument("address", help="Node address")
    
    remove_node_parser = subparsers.add_parser("remove-node", help="Remove a node")
    remove_node_parser.add_argument("node_id", help="Node ID")
    
    node_health_parser = subparsers.add_parser("node-health", help="Get node health")
    node_health_parser.add_argument("node_id", help="Node ID")
    
    # Shard commands
    shards_parser = subparsers.add_parser("shards", help="List shards")
    shards_parser.add_argument("--database", help="Filter by database ID")
    
    migrate_parser = subparsers.add_parser("migrate-shard", help="Migrate a shard")
    migrate_parser.add_argument("shard_id", help="Shard ID")
    migrate_parser.add_argument("target_node", help="Target node ID")
    
    shard_status_parser = subparsers.add_parser("shard-status", help="Get shard status")
    shard_status_parser.add_argument("shard_id", help="Shard ID")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = ClusterCLI(args.host, args.port)
    
    # Execute command
    result = None
    if args.command == "status":
        result = cli.cluster_status()
    elif args.command == "diagnostics":
        result = cli.diagnostics()
    elif args.command == "metrics":
        result = cli.metrics()
    elif args.command == "nodes":
        result = cli.list_nodes()
    elif args.command == "add-node":
        result = cli.add_node(args.node_id, args.address)
    elif args.command == "remove-node":
        result = cli.remove_node(args.node_id)
    elif args.command == "node-health":
        result = cli.node_health(args.node_id)
    elif args.command == "shards":
        result = cli.list_shards(getattr(args, "database", None))
    elif args.command == "migrate-shard":
        result = cli.migrate_shard(args.shard_id, args.target_node)
    elif args.command == "shard-status":
        result = cli.shard_status(args.shard_id)
    
    # Output result
    if result:
        print(format_output(result, args.format))
        if isinstance(result, dict) and "error" in result:
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
