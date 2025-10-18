#!/bin/bash

# Comprehensive security hardening script for JadeVectorDB
# This script implements advanced security features beyond basic authentication

set -e  # Exit on any error

echo "Implementing comprehensive security hardening for JadeVectorDB..."

# Function to print usage
print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --tls-setup             Setup TLS/SSL encryption"
    echo "  --rbac-setup            Setup Role-Based Access Control"
    echo "  --audit-setup           Setup comprehensive audit logging"
    echo "  --rate-limit-setup      Setup advanced rate limiting"
    echo "  --all                   Run all security hardening (default)"
    echo "  --help                  Show this help message"
}

# Parse command line options
TLS_SETUP=false
RBAC_SETUP=false
AUDIT_SETUP=false
RATE_LIMIT_SETUP=false
ALL=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --tls-setup)
            TLS_SETUP=true
            ALL=false
            shift
            ;;
        --rbac-setup)
            RBAC_SETUP=true
            ALL=false
            shift
            ;;
        --audit-setup)
            AUDIT_SETUP=true
            ALL=false
            shift
            ;;
        --rate-limit-setup)
            RATE_LIMIT_SETUP=true
            ALL=false
            shift
            ;;
        --all)
            ALL=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Define paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECURITY_CONFIG_DIR="$PROJECT_ROOT/security_configs"
mkdir -p "$SECURITY_CONFIG_DIR"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Setup TLS/SSL encryption
if [ "$RBAC_SETUP" = false ] && [ "$AUDIT_SETUP" = false ] && [ "$RATE_LIMIT_SETUP" = false ] && [ "$ALL" = true ] || [ "$TLS_SETUP" = true ]; then
    echo "Setting up TLS/SSL encryption..."
    
    # Check if openssl is available
    if ! command_exists openssl; then
        echo "WARNING: openssl not found. Install with: sudo apt install openssl"
    else
        echo "Generating self-signed TLS certificates..."
        openssl version
        
        # Create certificate directory
        CERT_DIR="$SECURITY_CONFIG_DIR/certs"
        mkdir -p "$CERT_DIR"
        
        # Generate private key
        echo "Generating private key..."
        openssl genrsa -out "$CERT_DIR/jadevectordb.key" 2048
        
        # Generate certificate signing request
        echo "Generating certificate signing request..."
        openssl req -new -key "$CERT_DIR/jadevectordb.key" -out "$CERT_DIR/jadevectordb.csr" -subj "/C=US/ST=State/L=City/O=JadeVectorDB/CN=localhost"
        
        # Generate self-signed certificate
        echo "Generating self-signed certificate..."
        openssl x509 -req -days 365 -in "$CERT_DIR/jadevectordb.csr" -signkey "$CERT_DIR/jadevectordb.key" -out "$CERT_DIR/jadevectordb.crt"
        
        # Set proper permissions
        chmod 600 "$CERT_DIR/jadevectordb.key"
        chmod 644 "$CERT_DIR/jadevectordb.crt"
        
        echo "TLS certificates generated in: $CERT_DIR"
        echo "To enable TLS, configure the service with:"
        echo "  JADE_DB_TLS_ENABLED=true"
        echo "  JADE_DB_TLS_CERT_FILE=$CERT_DIR/jadevectordb.crt"
        echo "  JADE_DB_TLS_KEY_FILE=$CERT_DIR/jadevectordb.key"
    fi
    
    # Create TLS configuration file
    cat > "$SECURITY_CONFIG_DIR/tls_config.json" << EOF
{
  "tls": {
    "enabled": true,
    "certificate_file": "$SECURITY_CONFIG_DIR/certs/jadevectordb.crt",
    "private_key_file": "$SECURITY_CONFIG_DIR/certs/jadevectordb.key",
    "ca_certificate_file": "",
    "client_auth": "none",
    "min_version": "TLS1.2",
    "cipher_suites": [
      "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
      "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
    ]
  }
}
EOF
    
    echo "TLS configuration file created: $SECURITY_CONFIG_DIR/tls_config.json"
fi

# Setup Role-Based Access Control (RBAC)
if [ "$TLS_SETUP" = false ] && [ "$AUDIT_SETUP" = false ] && [ "$RATE_LIMIT_SETUP" = false ] && [ "$ALL" = true ] || [ "$RBAC_SETUP" = true ]; then
    echo "Setting up Role-Based Access Control (RBAC)..."
    
    # Create RBAC configuration files
    cat > "$SECURITY_CONFIG_DIR/rbac_roles.json" << EOF
{
  "roles": {
    "admin": {
      "permissions": [
        "database:create",
        "database:read",
        "database:update",
        "database:delete",
        "database:list",
        "vector:add",
        "vector:read",
        "vector:update",
        "vector:delete",
        "search:execute",
        "index:create",
        "index:read",
        "index:update",
        "index:delete",
        "monitoring:read",
        "monitoring:write",
        "user:manage",
        "role:assign"
      ]
    },
    "developer": {
      "permissions": [
        "database:create",
        "database:read",
        "database:update",
        "database:delete",
        "database:list",
        "vector:add",
        "vector:read",
        "vector:update",
        "vector:delete",
        "search:execute",
        "index:create",
        "index:read",
        "index:update",
        "index:delete"
      ]
    },
    "analyst": {
      "permissions": [
        "database:read",
        "database:list",
        "vector:read",
        "search:execute",
        "index:read"
      ]
    },
    "viewer": {
      "permissions": [
        "database:read",
        "database:list",
        "vector:read",
        "search:execute"
      ]
    }
  }
}
EOF
    
    echo "RBAC roles configuration created: $SECURITY_CONFIG_DIR/rbac_roles.json"
    
    # Create sample users file
    cat > "$SECURITY_CONFIG_DIR/users.json" << EOF
{
  "users": {
    "admin_user": {
      "password_hash": "\$2b\$12\$example_hash_for_admin_password",
      "roles": ["admin"],
      "active": true,
      "created_at": "2023-01-01T00:00:00Z",
      "last_login": "2023-01-01T00:00:00Z"
    },
    "dev_user": {
      "password_hash": "\$2b\$12\$example_hash_for_dev_password",
      "roles": ["developer"],
      "active": true,
      "created_at": "2023-01-01T00:00:00Z",
      "last_login": "2023-01-01T00:00:00Z"
    },
    "analyst_user": {
      "password_hash": "\$2b\$12\$example_hash_for_analyst_password",
      "roles": ["analyst"],
      "active": true,
      "created_at": "2023-01-01T00:00:00Z",
      "last_login": "2023-01-01T00:00:00Z"
    }
  }
}
EOF
    
    echo "Sample users configuration created: $SECURITY_CONFIG_DIR/users.json"
    
    # Create RBAC policy file
    cat > "$SECURITY_CONFIG_DIR/rbac_policy.json" << EOF
{
  "policy": {
    "version": "1.0",
    "statements": [
      {
        "effect": "allow",
        "actions": ["database:*"],
        "resources": ["databases/*"],
        "principals": ["roles/admin"]
      },
      {
        "effect": "allow",
        "actions": ["database:create", "database:read", "database:update", "database:delete", "database:list"],
        "resources": ["databases/*"],
        "principals": ["roles/developer"]
      },
      {
        "effect": "allow",
        "actions": ["database:read", "database:list"],
        "resources": ["databases/*"],
        "principals": ["roles/analyst"]
      },
      {
        "effect": "allow",
        "actions": ["database:read", "database:list"],
        "resources": ["databases/*"],
        "principals": ["roles/viewer"]
      },
      {
        "effect": "allow",
        "actions": ["vector:*"],
        "resources": ["databases/*/vectors/*"],
        "principals": ["roles/admin", "roles/developer"]
      },
      {
        "effect": "allow",
        "actions": ["vector:read"],
        "resources": ["databases/*/vectors/*"],
        "principals": ["roles/analyst", "roles/viewer"]
      },
      {
        "effect": "allow",
        "actions": ["search:execute"],
        "resources": ["databases/*/search*"],
        "principals": ["roles/admin", "roles/developer", "roles/analyst", "roles/viewer"]
      },
      {
        "effect": "allow",
        "actions": ["index:*"],
        "resources": ["databases/*/indexes/*"],
        "principals": ["roles/admin", "roles/developer"]
      },
      {
        "effect": "allow",
        "actions": ["index:read"],
        "resources": ["databases/*/indexes/*"],
        "principals": ["roles/analyst"]
      }
    ]
  }
}
EOF
    
    echo "RBAC policy configuration created: $SECURITY_CONFIG_DIR/rbac_policy.json"
fi

# Setup comprehensive audit logging
if [ "$TLS_SETUP" = false ] && [ "$RBAC_SETUP" = false ] && [ "$RATE_LIMIT_SETUP" = false ] && [ "$ALL" = true ] || [ "$AUDIT_SETUP" = true ]; then
    echo "Setting up comprehensive audit logging..."
    
    # Create audit logging configuration
    cat > "$SECURITY_CONFIG_DIR/audit_config.json" << EOF
{
  "audit_logging": {
    "enabled": true,
    "log_file": "$SECURITY_CONFIG_DIR/logs/audit.log",
    "rotation": {
      "max_size_mb": 100,
      "max_files": 10,
      "compress": true
    },
    "events": {
      "authentication": {
        "login_attempts": true,
        "failed_logins": true,
        "logout_events": true
      },
      "authorization": {
        "access_denied": true,
        "permission_checks": true
      },
      "data_operations": {
        "vector_create": true,
        "vector_read": true,
        "vector_update": true,
        "vector_delete": true,
        "database_create": true,
        "database_delete": true,
        "search_operations": true
      },
      "system_operations": {
        "configuration_changes": true,
        "service_start_stop": true,
        "index_operations": true
      }
    },
    "sensitive_data_masking": {
      "enabled": true,
      "mask_fields": ["password", "api_key", "secret", "token"],
      "mask_character": "*"
    }
  }
}
EOF
    
    echo "Audit logging configuration created: $SECURITY_CONFIG_DIR/audit_config.json"
    
    # Create audit log directory
    mkdir -p "$SECURITY_CONFIG_DIR/logs"
    touch "$SECURITY_CONFIG_DIR/logs/audit.log"
    
    # Create log rotation script
    cat > "$SECURITY_CONFIG_DIR/scripts/rotate_audit_logs.sh" << 'EOF'
#!/bin/bash

# Simple log rotation script for audit logs
LOG_FILE="$1"
MAX_SIZE_MB=${2:-100}
MAX_FILES=${3:-10}

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file $LOG_FILE does not exist"
    exit 1
fi

# Check file size
FILE_SIZE_MB=$(du -m "$LOG_FILE" | cut -f1)

if [ "$FILE_SIZE_MB" -ge "$MAX_SIZE_MB" ]; then
    # Rotate logs
    for i in $(seq $((MAX_FILES - 1)) -1 1); do
        if [ -f "${LOG_FILE}.${i}" ]; then
            mv "${LOG_FILE}.${i}" "${LOG_FILE}.$((i + 1))"
        fi
    done
    
    # Move current log to .1
    mv "$LOG_FILE" "${LOG_FILE}.1"
    
    # Create new log file
    touch "$LOG_FILE"
    
    echo "Log rotated: $LOG_FILE"
fi
EOF
    
    chmod +x "$SECURITY_CONFIG_DIR/scripts/rotate_audit_logs.sh"
    echo "Audit log rotation script created: $SECURITY_CONFIG_DIR/scripts/rotate_audit_logs.sh"
fi

# Setup advanced rate limiting
if [ "$TLS_SETUP" = false ] && [ "$RBAC_SETUP" = false ] && [ "$AUDIT_SETUP" = false ] && [ "$ALL" = true ] || [ "$RATE_LIMIT_SETUP" = true ]; then
    echo "Setting up advanced rate limiting..."
    
    # Create rate limiting configuration
    cat > "$SECURITY_CONFIG_DIR/rate_limit_config.json" << EOF
{
  "rate_limiting": {
    "enabled": true,
    "default_limits": {
      "requests_per_minute": 1000,
      "requests_per_hour": 10000
    },
    "per_endpoint_limits": {
      "/v1/databases": {
        "requests_per_minute": 100,
        "requests_per_hour": 1000
      },
      "/v1/databases/*/vectors": {
        "requests_per_minute": 1000,
        "requests_per_hour": 10000
      },
      "/v1/databases/*/search": {
        "requests_per_minute": 500,
        "requests_per_hour": 5000
      },
      "/v1/embeddings/generate": {
        "requests_per_minute": 10,
        "requests_per_hour": 100
      }
    },
    "ip_whitelist": [
      "127.0.0.1",
      "::1"
    ],
    "burst_allowance": {
      "enabled": true,
      "multiplier": 2.0,
      "duration_seconds": 60
    },
    "adaptive_throttling": {
      "enabled": true,
      "error_rate_threshold": 0.1,
      "adjustment_factor": 0.5,
      "cooldown_period_seconds": 300
    }
  }
}
EOF
    
    echo "Rate limiting configuration created: $SECURITY_CONFIG_DIR/rate_limit_config.json"
    
    # Create IP blocking configuration
    cat > "$SECURITY_CONFIG_DIR/ip_blocking_config.json" << EOF
{
  "ip_blocking": {
    "enabled": true,
    "blocked_ips": [],
    "automatic_blocking": {
      "enabled": true,
      "failed_login_threshold": 5,
      "block_duration_minutes": 30,
      "reset_after_minutes": 60
    },
    "geo_blocking": {
      "enabled": false,
      "blocked_countries": []
    }
  }
}
EOF
    
    echo "IP blocking configuration created: $SECURITY_CONFIG_DIR/ip_blocking_config.json"
fi

# Generate security hardening report
echo
echo "==============================================="
echo "SECURITY HARDENING REPORT"
echo "==============================================="

if [ "$TLS_SETUP" = false ] && [ "$RBAC_SETUP" = false ] && [ "$AUDIT_SETUP" = false ] && [ "$RATE_LIMIT_SETUP" = false ] && [ "$ALL" = true ]; then
    echo "✓ TLS/SSL encryption setup completed"
    echo "✓ Role-Based Access Control (RBAC) configured"
    echo "✓ Comprehensive audit logging implemented"
    echo "✓ Advanced rate limiting configured"
elif [ "$TLS_SETUP" = true ]; then
    echo "✓ TLS/SSL encryption setup only completed"
elif [ "$RBAC_SETUP" = true ]; then
    echo "✓ Role-Based Access Control (RBAC) only configured"
elif [ "$AUDIT_SETUP" = true ]; then
    echo "✓ Comprehensive audit logging only implemented"
elif [ "$RATE_LIMIT_SETUP" = true ]; then
    echo "✓ Advanced rate limiting only configured"
fi

echo "==============================================="
echo "Security configurations generated in: $SECURITY_CONFIG_DIR"
echo "Review all configuration files and implement in your deployment"
echo "==============================================="

echo "Comprehensive security hardening completed!"