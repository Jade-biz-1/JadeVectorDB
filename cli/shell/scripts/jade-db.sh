#!/bin/bash

# JadeVectorDB Shell CLI
# A shell script interface for interacting with the JadeVectorDB API

set -e

# Default configuration
DEFAULT_URL="http://localhost:8080"
API_KEY=""
DATABASE_ID=""
VECTOR_ID=""
QUERY_VECTOR=""
TOP_K=10
CURL_ONLY=false

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS] COMMAND [ARGS...]"
    echo ""
    echo "Options:"
    echo "  --url URL          JadeVectorDB API URL (default: $DEFAULT_URL)"
    echo "  --api-key KEY      API key for authentication"
    echo "  --curl-only        Generate cURL commands instead of executing"
    echo ""
    echo "Commands:"
    echo "  create-db NAME [DESCRIPTION] [DIMENSION] [INDEX_TYPE]    Create a new database"
    echo "  list-dbs                                                 List all databases"
    echo "  get-db ID                                                Get database info"
    echo "  store ID VALUES [METADATA]                               Store a vector"
    echo "  retrieve ID                                              Retrieve a vector"
    echo "  delete ID                                                Delete a vector"
    echo "  search QUERY_VECTOR [TOP_K] [THRESHOLD]                  Search for similar vectors"
    echo "  status                                                   Get system status"
    echo "  health                                                   Get system health"
    echo ""
    echo "Examples:"
    echo "  $0 --url http://localhost:8080 list-dbs"
    echo "  $0 --api-key mykey123 store myvector '[0.1, 0.2, 0.3]' '{\"category\":\"test\"}'"
    echo "  $0 --curl-only --url http://localhost:8080 list-dbs"
    echo ""
    exit 1
}

# Function to make API call
api_call() {
    local method=$1
    local endpoint=$2
    local data=$3

    local auth_header=""
    if [ -n "$API_KEY" ]; then
        auth_header="Authorization: Bearer $API_KEY"
    fi

    if [ "$CURL_ONLY" = true ]; then
        echo "# cURL command for $method $endpoint"
        echo "curl -s -X $method \\"
        echo "  -H \"Content-Type: application/json\" \\"
        if [ -n "$auth_header" ]; then
            echo "  -H \"$auth_header\" \\"
        fi
        if [ -n "$data" ] && [ "$method" != "GET" ] && [ "$method" != "DELETE" ]; then
            echo "  -d '$data' \\"
        fi
        echo "  \"$BASE_URL$endpoint\""
        return
    fi

    if [ "$method" = "GET" ]; then
        curl -s -X GET \
            -H "Content-Type: application/json" \
            -H "$auth_header" \
            "$BASE_URL$endpoint"
    elif [ "$method" = "POST" ]; then
        curl -s -X POST \
            -H "Content-Type: application/json" \
            -H "$auth_header" \
            -d "$data" \
            "$BASE_URL$endpoint"
    elif [ "$method" = "DELETE" ]; then
        curl -s -X DELETE \
            -H "Content-Type: application/json" \
            -H "$auth_header" \
            "$BASE_URL$endpoint"
    fi
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --url)
            BASE_URL="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --curl-only)
            CURL_ONLY=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                COMMAND="$1"
                shift
            else
                # Collect all remaining arguments as command arguments
                break
            fi
            ;;
    esac
done

# Set default base URL if not provided
if [ -z "$BASE_URL" ]; then
    BASE_URL="$DEFAULT_URL"
fi

# Execute command
case "$COMMAND" in
    create-db)
        if [ -z "$2" ]; then
            echo "Error: Database name is required"
            usage
        fi
        
        DB_NAME="$2"
        DESCRIPTION="${3:-""}"
        DIMENSION="${4:-128}"
        INDEX_TYPE="${5:-"HNSW"}"
        
        if [ "$CURL_ONLY" = true ]; then
            echo "# cURL command for creating database: $DB_NAME"
            echo "curl -X POST \\"
            echo "  -H \"Content-Type: application/json\" \\"
            if [ -n "$API_KEY" ]; then
                echo "  -H \"Authorization: Bearer $API_KEY\" \\"
            fi
            echo "  -d '{"
            echo "    \"name\": \"$DB_NAME\","
            echo "    \"description\": \"$DESCRIPTION\","
            echo "    \"vectorDimension\": $DIMENSION,"
            echo "    \"indexType\": \"$INDEX_TYPE\""
            echo "  }' \\"
            echo "  \"$BASE_URL/v1/databases\""
        else
            DATA=$(cat <<EOF
{
    "name": "$DB_NAME",
    "description": "$DESCRIPTION",
    "vectorDimension": $DIMENSION,
    "indexType": "$INDEX_TYPE"
}
EOF
)
            result=$(api_call "POST" "/v1/databases" "$DATA")
            echo "$result"
        fi
        ;;
    list-dbs)
        if [ "$CURL_ONLY" = true ]; then
            echo "# cURL command for listing databases"
            echo "curl -X GET \\"
            echo "  -H \"Content-Type: application/json\" \\"
            if [ -n "$API_KEY" ]; then
                echo "  -H \"Authorization: Bearer $API_KEY\" \\"
            fi
            echo "  \"$BASE_URL/v1/databases\""
        else
            result=$(api_call "GET" "/v1/databases" "")
            echo "$result"
        fi
        ;;
    get-db)
        if [ -z "$2" ]; then
            echo "Error: Database ID is required"
            usage
        fi
        
        DB_ID="$2"
        if [ "$CURL_ONLY" = true ]; then
            echo "# cURL command for getting database: $DB_ID"
            echo "curl -X GET \\"
            echo "  -H \"Content-Type: application/json\" \\"
            if [ -n "$API_KEY" ]; then
                echo "  -H \"Authorization: Bearer $API_KEY\" \\"
            fi
            echo "  \"$BASE_URL/v1/databases/$DB_ID\""
        else
            result=$(api_call "GET" "/v1/databases/$DB_ID" "")
            echo "$result"
        fi
        ;;
    store)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Error: Vector ID and values are required"
            usage
        fi
        
        VECTOR_ID="$2"
        VALUES="$3"
        METADATA="${4:-""}"
        
        if [ "$CURL_ONLY" = true ]; then
            echo "# cURL command for storing vector: $VECTOR_ID"
            echo "curl -X POST \\"
            echo "  -H \"Content-Type: application/json\" \\"
            if [ -n "$API_KEY" ]; then
                echo "  -H \"Authorization: Bearer $API_KEY\" \\"
            fi
            if [ -n "$METADATA" ] && [ "$METADATA" != "" ]; then
                echo "  -d '{"
                echo "    \"id\": \"$VECTOR_ID\","
                echo "    \"values\": $VALUES,"
                echo "    \"metadata\": $METADATA"
                echo "  }' \\"
            else
                echo "  -d '{"
                echo "    \"id\": \"$VECTOR_ID\","
                echo "    \"values\": $VALUES"
                echo "  }' \\"
            fi
            echo "  \"$BASE_URL/v1/databases/$DATABASE_ID/vectors\""
        else
            # Prepare data JSON
            if [ -n "$METADATA" ] && [ "$METADATA" != "" ]; then
                DATA=$(cat <<EOF
{
    "id": "$VECTOR_ID",
    "values": $VALUES,
    "metadata": $METADATA
}
EOF
)
            else
                DATA=$(cat <<EOF
{
    "id": "$VECTOR_ID",
    "values": $VALUES
}
EOF
)
            fi
            
            result=$(api_call "POST" "/v1/databases/$DATABASE_ID/vectors" "$DATA")
            echo "$result"
        fi
        ;;
    retrieve)
        if [ -z "$2" ]; then
            echo "Error: Vector ID is required"
            usage
        fi
        
        VECTOR_ID="$2"
        if [ "$CURL_ONLY" = true ]; then
            echo "# cURL command for retrieving vector: $VECTOR_ID"
            echo "curl -X GET \\"
            echo "  -H \"Content-Type: application/json\" \\"
            if [ -n "$API_KEY" ]; then
                echo "  -H \"Authorization: Bearer $API_KEY\" \\"
            fi
            echo "  \"$BASE_URL/v1/databases/$DATABASE_ID/vectors/$VECTOR_ID\""
        else
            result=$(api_call "GET" "/v1/databases/$DATABASE_ID/vectors/$VECTOR_ID" "")
            echo "$result"
        fi
        ;;
    delete)
        if [ -z "$2" ]; then
            echo "Error: Vector ID is required"
            usage
        fi
        
        VECTOR_ID="$2"
        if [ "$CURL_ONLY" = true ]; then
            echo "# cURL command for deleting vector: $VECTOR_ID"
            echo "curl -X DELETE \\"
            echo "  -H \"Content-Type: application/json\" \\"
            if [ -n "$API_KEY" ]; then
                echo "  -H \"Authorization: Bearer $API_KEY\" \\"
            fi
            echo "  \"$BASE_URL/v1/databases/$DATABASE_ID/vectors/$VECTOR_ID\""
        else
            result=$(api_call "DELETE" "/v1/databases/$DATABASE_ID/vectors/$VECTOR_ID" "")
            echo "$result"
        fi
        ;;
    search)
        if [ -z "$2" ]; then
            echo "Error: Query vector is required"
            usage
        fi
        
        QUERY_VECTOR="$2"
        TOP_K="${3:-10}"
        THRESHOLD="${4:-""}"
        
        if [ "$CURL_ONLY" = true ]; then
            echo "# cURL command for similarity search"
            echo "curl -X POST \\"
            echo "  -H \"Content-Type: application/json\" \\"
            if [ -n "$API_KEY" ]; then
                echo "  -H \"Authorization: Bearer $API_KEY\" \\"
            fi
            if [ -n "$THRESHOLD" ] && [ "$THRESHOLD" != "" ]; then
                echo "  -d '{"
                echo "    \"queryVector\": $QUERY_VECTOR,"
                echo "    \"topK\": $TOP_K,"
                echo "    \"threshold\": $THRESHOLD"
                echo "  }' \\"
            else
                echo "  -d '{"
                echo "    \"queryVector\": $QUERY_VECTOR,"
                echo "    \"topK\": $TOP_K"
                echo "  }' \\"
            fi
            echo "  \"$BASE_URL/v1/databases/$DATABASE_ID/search\""
        else
            # Prepare data JSON
            if [ -n "$THRESHOLD" ] && [ "$THRESHOLD" != "" ]; then
                DATA=$(cat <<EOF
{
    "queryVector": $QUERY_VECTOR,
    "topK": $TOP_K,
    "threshold": $THRESHOLD
}
EOF
)
            else
                DATA=$(cat <<EOF
{
    "queryVector": $QUERY_VECTOR,
    "topK": $TOP_K
}
EOF
)
            fi
            
            result=$(api_call "POST" "/v1/databases/$DATABASE_ID/search" "$DATA")
            echo "$result"
        fi
        ;;
    status)
        if [ "$CURL_ONLY" = true ]; then
            echo "# cURL command for getting system status"
            echo "curl -X GET \\"
            echo "  -H \"Content-Type: application/json\" \\"
            if [ -n "$API_KEY" ]; then
                echo "  -H \"Authorization: Bearer $API_KEY\" \\"
            fi
            echo "  \"$BASE_URL/status\""
        else
            result=$(api_call "GET" "/status" "")
            echo "$result"
        fi
        ;;
    health)
        if [ "$CURL_ONLY" = true ]; then
            echo "# cURL command for getting system health"
            echo "curl -X GET \\"
            echo "  -H \"Content-Type: application/json\" \\"
            if [ -n "$API_KEY" ]; then
                echo "  -H \"Authorization: Bearer $API_KEY\" \\"
            fi
            echo "  \"$BASE_URL/health\""
        else
            result=$(api_call "GET" "/health" "")
            echo "$result"
        fi
        ;;
    "")
        echo "Error: No command specified"
        usage
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        usage
        ;;
esac