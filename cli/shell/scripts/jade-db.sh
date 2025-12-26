#!/bin/bash

# JadeVectorDB Shell CLI
# A shell script interface for interacting with the JadeVectorDB API

set -e

# Default configuration
# Support environment variables
DEFAULT_URL="${JADEVECTORDB_URL:-http://localhost:8080}"
API_KEY="${JADEVECTORDB_API_KEY:-}"
DATABASE_ID="${JADEVECTORDB_DATABASE_ID:-}"
VECTOR_ID=""
QUERY_VECTOR=""
TOP_K=10
CURL_ONLY=false
OUTPUT_FORMAT="json"

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS] COMMAND [ARGS...]"
    echo ""
    echo "Options:"
    echo "  --url URL            JadeVectorDB API URL (default: $DEFAULT_URL)"
    echo "                       Can also be set via JADEVECTORDB_URL environment variable"
    echo "  --api-key KEY        API key for authentication"
    echo "                       Can also be set via JADEVECTORDB_API_KEY environment variable"
    echo "  --database-id ID     Database ID (required for vector operations)"
    echo "                       Can also be set via JADEVECTORDB_DATABASE_ID environment variable"
    echo "  --curl-only          Generate cURL commands instead of executing"
    echo "  --format FORMAT      Output format: json (default), yaml, table, csv"
    echo ""
    echo "Commands:"
    echo "  Database Operations:"
    echo "    create-db NAME [DESCRIPTION] [DIMENSION] [INDEX_TYPE]    Create a new database"
    echo "    list-dbs                                                 List all databases"
    echo "    get-db ID                                                Get database info"
    echo "    delete-db ID                                             Delete a database"
    echo ""
    echo "  Vector Operations:"
    echo "    store ID VALUES [METADATA]                               Store a vector (requires --database-id)"
    echo "    retrieve ID                                              Retrieve a vector (requires --database-id)"
    echo "    delete ID                                                Delete a vector (requires --database-id)"
    echo "    search QUERY_VECTOR [TOP_K] [THRESHOLD]                  Search for similar vectors (requires --database-id)"
    echo ""
    echo "  User Management:"
    echo "    user-add USERNAME ROLE [PASSWORD] [EMAIL]               Add a new user"
    echo "    user-list [ROLE] [STATUS]                                List all users"
    echo "    user-show USER_ID                                        Show user details"
    echo "    user-update USER_ID [ROLE] [STATUS]                      Update user info"
    echo "    user-delete USER_ID                                      Delete a user"
    echo "    user-activate USER_ID                                    Activate a user"
    echo "    user-deactivate USER_ID                                  Deactivate a user"
    echo ""
    echo "  Import/Export:"
    echo "    import FILE DATABASE_ID                                  Import vectors from JSON file"
    echo "    export FILE DATABASE_ID [VECTOR_IDS]                     Export vectors to JSON file"
    echo ""
    echo "  System:"
    echo "    status                                                   Get system status"
    echo "    health                                                   Get system health"
    echo ""
    echo "Examples:"
    echo "  $0 --url http://localhost:8080 list-dbs"
    echo "  $0 --database-id mydb --api-key mykey123 store myvector '[0.1, 0.2, 0.3]' '{\"category\":\"test\"}'"
    echo "  $0 --curl-only --url http://localhost:8080 list-dbs"
    echo ""
    exit 1
}

# Function to format output
format_output() {
    local data="$1"
    local format="${2:-json}"

    case "$format" in
        json)
            echo "$data" | jq '.' 2>/dev/null || echo "$data"
            ;;
        yaml)
            if command -v yq &> /dev/null; then
                echo "$data" | yq eval -P 2>/dev/null || echo "$data"
            else
                echo "Warning: yq not installed. Install with: brew install yq (Mac) or snap install yq (Linux)" >&2
                echo "Falling back to JSON output:" >&2
                echo "$data" | jq '.' 2>/dev/null || echo "$data"
            fi
            ;;
        table)
            if echo "$data" | jq -e 'type == "array"' &> /dev/null; then
                # Array of objects - format as table
                echo "$data" | jq -r '(.[0] | keys_unsorted) as $keys | $keys, map([.[ $keys[] ]])[] | @tsv' | column -t -s $'\t' 2>/dev/null || echo "$data"
            elif echo "$data" | jq -e 'type == "object"' &> /dev/null; then
                # Single object - format as key-value table
                echo "$data" | jq -r 'to_entries | .[] | [.key, .value] | @tsv' | column -t -s $'\t' 2>/dev/null || echo "$data"
            else
                echo "$data"
            fi
            ;;
        csv)
            if echo "$data" | jq -e 'type == "array"' &> /dev/null; then
                # Array of objects - format as CSV
                echo "$data" | jq -r '(.[0] | keys_unsorted) as $keys | ($keys | @csv), (map([.[ $keys[] ]]) | .[] | @csv)' 2>/dev/null || echo "$data"
            elif echo "$data" | jq -e 'type == "object"' &> /dev/null; then
                # Single object - format as CSV with key-value pairs
                echo "Key,Value"
                echo "$data" | jq -r 'to_entries | .[] | [.key, .value] | @csv' 2>/dev/null || echo "$data"
            else
                echo "$data"
            fi
            ;;
        *)
            echo "Error: Unsupported format: $format" >&2
            echo "$data"
            ;;
    esac
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
        if [ -n "$auth_header" ]; then
            curl -s -X GET \
                -H "Content-Type: application/json" \
                -H "$auth_header" \
                "$BASE_URL$endpoint"
        else
            curl -s -X GET \
                -H "Content-Type: application/json" \
                "$BASE_URL$endpoint"
        fi
    elif [ "$method" = "POST" ]; then
        if [ -n "$auth_header" ]; then
            curl -s -X POST \
                -H "Content-Type: application/json" \
                -H "$auth_header" \
                -d "$data" \
                "$BASE_URL$endpoint"
        else
            curl -s -X POST \
                -H "Content-Type: application/json" \
                -d "$data" \
                "$BASE_URL$endpoint"
        fi
    elif [ "$method" = "DELETE" ]; then
        if [ -n "$auth_header" ]; then
            curl -s -X DELETE \
                -H "Content-Type: application/json" \
                -H "$auth_header" \
                "$BASE_URL$endpoint"
        else
            curl -s -X DELETE \
                -H "Content-Type: application/json" \
                "$BASE_URL$endpoint"
        fi
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
        --database-id)
            DATABASE_ID="$2"
            shift 2
            ;;
        --curl-only)
            CURL_ONLY=true
            shift
            ;;
        --format)
            OUTPUT_FORMAT="$2"
            shift 2
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
        if [ -z "$1" ]; then
            echo "Error: Database name is required"
            usage
        fi

        DB_NAME="$1"
        DESCRIPTION="${2:-""}"
        DIMENSION="${3:-128}"
        INDEX_TYPE="${4:-hnsw}"
        
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
            format_output "$result" "$OUTPUT_FORMAT"
        fi
        ;;
    get-db)
        if [ -z "$1" ]; then
            echo "Error: Database ID is required"
            usage
        fi

        DB_ID="$1"
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
    delete-db)
        if [ -z "$1" ]; then
            echo "Error: Database ID is required"
            usage
        fi

        DB_ID="$1"
        if [ "$CURL_ONLY" = true ]; then
            echo "# cURL command for deleting database: $DB_ID"
            echo "curl -X DELETE \\"
            echo "  -H \"Content-Type: application/json\" \\"
            if [ -n "$API_KEY" ]; then
                echo "  -H \"Authorization: Bearer $API_KEY\" \\"
            fi
            echo "  \"$BASE_URL/v1/databases/$DB_ID\""
        else
            result=$(api_call "DELETE" "/v1/databases/$DB_ID" "")
            echo "$result"
        fi
        ;;
    store)
        if [ -z "$1" ] || [ -z "$2" ]; then
            echo "Error: Vector ID and values are required"
            usage
        fi

        VECTOR_ID="$1"
        VALUES="$2"
        METADATA="${3:-""}"
        
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
        if [ -z "$1" ]; then
            echo "Error: Vector ID is required"
            usage
        fi
        
        VECTOR_ID="$1"
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
        if [ -z "$1" ]; then
            echo "Error: Vector ID is required"
            usage
        fi
        
        VECTOR_ID="$1"
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
        if [ -z "$1" ]; then
            echo "Error: Query vector is required"
            usage
        fi
        
        QUERY_VECTOR="$1"
        TOP_K="${2:-10}"
        THRESHOLD="${3:-""}"
        
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
    user-add)
        if [ -z "$1" ] || [ -z "$2" ]; then
            echo "Error: Username and role are required"
            usage
        fi

        USERNAME="$1"
        ROLE="$2"
        PASSWORD="${3:-""}"
        EMAIL="${4:-""}"

        if [ -n "$EMAIL" ]; then
            DATA="{\"username\":\"$USERNAME\",\"roles\":[\"$ROLE\"],\"password\":\"$PASSWORD\",\"email\":\"$EMAIL\"}"
        elif [ -n "$PASSWORD" ]; then
            DATA="{\"username\":\"$USERNAME\",\"roles\":[\"$ROLE\"],\"password\":\"$PASSWORD\"}"
        else
            DATA="{\"username\":\"$USERNAME\",\"roles\":[\"$ROLE\"]}"
        fi

        result=$(api_call "POST" "/v1/users" "$DATA")
        echo "$result"
        ;;
    user-list)
        ROLE_FILTER="${2:-}"
        STATUS_FILTER="${3:-}"

        QUERY_PARAMS=""
        if [ -n "$ROLE_FILTER" ]; then
            QUERY_PARAMS="?role=$ROLE_FILTER"
        fi
        if [ -n "$STATUS_FILTER" ]; then
            if [ -n "$QUERY_PARAMS" ]; then
                QUERY_PARAMS="${QUERY_PARAMS}&status=$STATUS_FILTER"
            else
                QUERY_PARAMS="?status=$STATUS_FILTER"
            fi
        fi

        result=$(api_call "GET" "/v1/users${QUERY_PARAMS}" "")
        echo "$result"
        ;;
    user-show)
        if [ -z "$1" ]; then
            echo "Error: User ID is required"
            usage
        fi

        USER_ID="$1"
        result=$(api_call "GET" "/v1/users/$USER_ID" "")
        echo "$result"
        ;;
    user-update)
        if [ -z "$1" ]; then
            echo "Error: User ID is required"
            usage
        fi

        USER_ID="$1"
        NEW_ROLE="${2:-}"
        NEW_STATUS="${3:-}"

        if [ -z "$NEW_ROLE" ] && [ -z "$NEW_STATUS" ]; then
            echo "Error: At least one of role or status must be provided"
            exit 1
        fi

        DATA="{"
        FIRST=true
        if [ -n "$NEW_ROLE" ]; then
            DATA="${DATA}\"roles\":[\"$NEW_ROLE\"]"
            FIRST=false
        fi
        if [ -n "$NEW_STATUS" ]; then
            if [ "$FIRST" = false ]; then
                DATA="${DATA},"
            fi
            # Convert status string to boolean is_active
            if [ "$NEW_STATUS" = "active" ]; then
                DATA="${DATA}\"is_active\":true"
            else
                DATA="${DATA}\"is_active\":false"
            fi
        fi
        DATA="${DATA}}"

        result=$(api_call "PUT" "/v1/users/$USER_ID" "$DATA")
        echo "$result"
        ;;
    user-delete)
        if [ -z "$1" ]; then
            echo "Error: User ID is required"
            usage
        fi

        USER_ID="$1"
        result=$(api_call "DELETE" "/v1/users/$USER_ID" "")
        echo "$result"
        ;;
    user-activate)
        if [ -z "$1" ]; then
            echo "Error: User ID is required"
            usage
        fi

        USER_ID="$1"
        DATA="{\"is_active\":true}"
        result=$(api_call "PUT" "/v1/users/$USER_ID" "$DATA")
        echo "$result"
        ;;
    user-deactivate)
        if [ -z "$1" ]; then
            echo "Error: User ID is required"
            usage
        fi

        USER_ID="$1"
        DATA="{\"is_active\":false}"
        result=$(api_call "PUT" "/v1/users/$USER_ID" "$DATA")
        echo "$result"
        ;;
    import)
        if [ -z "$1" ] || [ -z "$2" ]; then
            echo "Error: FILE and DATABASE_ID are required"
            usage
        fi

        FILE="$1"
        DB_ID="$2"

        if [ ! -f "$FILE" ]; then
            echo "Error: File not found: $FILE"
            exit 1
        fi

        echo "Importing vectors from $FILE to database $DB_ID..."

        # Read JSON file and import vectors one by one
        IMPORTED=0
        ERRORS=0

        # Use jq to parse JSON array and iterate
        while IFS= read -r vector_data; do
            VECTOR_ID=$(echo "$vector_data" | jq -r '.id')
            VALUES=$(echo "$vector_data" | jq -c '.values')
            METADATA=$(echo "$vector_data" | jq -c '.metadata // {}')

            DATA="{\"id\":\"$VECTOR_ID\",\"values\":$VALUES,\"metadata\":$METADATA}"

            result=$(api_call "POST" "/v1/databases/$DB_ID/vectors" "$DATA" 2>&1)
            if [ $? -eq 0 ]; then
                IMPORTED=$((IMPORTED + 1))
                echo -ne "\rImported: $IMPORTED vectors"
            else
                ERRORS=$((ERRORS + 1))
            fi
        done < <(jq -c '.[]' "$FILE")

        echo ""
        echo "Import completed: $IMPORTED vectors imported, $ERRORS errors"
        ;;
    export)
        if [ -z "$1" ] || [ -z "$2" ]; then
            echo "Error: FILE and DATABASE_ID are required"
            usage
        fi

        FILE="$1"
        DB_ID="$2"
        VECTOR_IDS="$3"

        echo "Exporting vectors from database $DB_ID to $FILE..."

        # If vector IDs provided, export those
        if [ -n "$VECTOR_IDS" ]; then
            echo "[" > "$FILE"
            FIRST=true

            IFS=',' read -ra IDS <<< "$VECTOR_IDS"
            for vid in "${IDS[@]}"; do
                result=$(api_call "GET" "/v1/databases/$DB_ID/vectors/$vid" "")
                if [ $? -eq 0 ]; then
                    if [ "$FIRST" = false ]; then
                        echo "," >> "$FILE"
                    fi
                    echo "$result" | jq -c '.' >> "$FILE"
                    FIRST=false
                fi
            done

            echo "]" >> "$FILE"
            echo "Export completed: ${#IDS[@]} vectors exported to $FILE"
        else
            echo "Note: For export, provide vector IDs as comma-separated list"
            echo "Usage: $0 export $FILE $DB_ID \"vec1,vec2,vec3\""
            exit 1
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