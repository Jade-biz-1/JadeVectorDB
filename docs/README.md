# JadeVectorDB Documentation

Welcome to the JadeVectorDB documentation. This directory contains comprehensive documentation for users, administrators, and developers.

**Last Updated**: December 26, 2025

---

## Quick Links

### Getting Started
- [Quick Start Guide](quickstart.md) - Get up and running in 5 minutes
- [Installation Guide](INSTALLATION_GUIDE.md) - Detailed installation instructions
- [User Guide](UserGuide.md) - Comprehensive user guide

### For Administrators
- [Operations Runbook](operations_runbook.md) - **NEW: Includes shutdown procedures**
- [Admin Endpoints Reference](admin_endpoints.md) - **NEW: Admin-only API endpoints**
- [RBAC Admin Guide](rbac_admin_guide.md) - Role-based access control
- [Configuration Guide](CONFIGURATION_GUIDE.md) - Server configuration
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md) - Common issues and solutions

### For Developers
- [API Documentation](api_documentation.md) - Complete API reference
- [API Reference](api/api_reference.md) - **NEW: Updated with admin endpoints**
- [Architecture](architecture.md) - System architecture overview
- [CLI Documentation](cli-documentation.md) - Command-line interface

### Deployment
- [Docker Deployment](DOCKER_DEPLOYMENT.md) - Deploy with Docker
- [Local Deployment](LOCAL_DEPLOYMENT.md) - Run locally
- [Distributed Deployment Guide](distributed_deployment_guide.md) - Multi-node setup
- [Kubernetes Guide](../k8s/README.md) - Deploy to Kubernetes

### Security
- [Security Policy](security_policy.md) - Security best practices
- [RBAC Permission Model](rbac_permission_model.md) - Permission system
- [Zero Trust Architecture](zero_trust_architecture.md) - Security architecture
- [Incident Response](incident_response.md) - Security incident handling

### Advanced Topics
- [Distributed Services API](distributed_services_api.md) - Cluster operations
- [Vector Compression](vector_compression.md) - Optimize storage
- [GPU Acceleration](gpu_acceleration.md) - GPU support
- [Predictive Maintenance](predictive_maintenance.md) - System maintenance
- [Advanced Embedding Models](advanced_embedding_models.md) - ML integration

### Operations
- [Monitoring Setup](monitoring_setup.md) - Configure monitoring
- [Backup and Restore](operations_runbook.md#backup-and-restore) - Data protection
- [Performance Tuning](operations_runbook.md#performance-tuning) - Optimize performance
- [Migration Guide](migration_guide_persistent_storage.md) - Data migration

### Testing
- [Test Execution Guide](TEST_EXECUTION_GUIDE.md) - Running tests
- [Security Testing Framework](security_testing_framework.md) - Security tests

---

## Recent Updates

### December 26, 2025 - Shutdown Feature

Added graceful server shutdown functionality:
- **NEW**: [Admin Endpoints Reference](admin_endpoints.md) - Complete guide for admin endpoints
- **NEW**: [Shutdown Feature Summary](SHUTDOWN_FEATURE.md) - Implementation details
- **UPDATED**: [Operations Runbook](operations_runbook.md) - Added shutdown procedures section
- **UPDATED**: [API Reference](api/api_reference.md) - Added admin shutdown endpoint

**Features**:
- POST /admin/shutdown endpoint for graceful server shutdown
- JWT authentication with admin role authorization
- Frontend dashboard button (visible only to admin users)
- Comprehensive documentation and troubleshooting guides

### Key Documentation Files

#### For Quick Reference
- **Shutdown Server**: See [Admin Endpoints](admin_endpoints.md#shutdown-endpoint) or [Operations Runbook](operations_runbook.md#shutdown-procedures)
- **User Management**: See [RBAC Admin Guide](rbac_admin_guide.md)
- **API Usage**: See [API Documentation](api_documentation.md)
- **Troubleshooting**: See [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)

---

## Documentation by Role

### System Administrator

**Essential Reading**:
1. [Operations Runbook](operations_runbook.md) - Daily operations
2. [Admin Endpoints Reference](admin_endpoints.md) - Admin tools
3. [RBAC Admin Guide](rbac_admin_guide.md) - User management
4. [Configuration Guide](CONFIGURATION_GUIDE.md) - System configuration

**Deployment & Scaling**:
- [Docker Deployment](DOCKER_DEPLOYMENT.md)
- [Distributed Deployment](distributed_deployment_guide.md)
- [Monitoring Setup](monitoring_setup.md)

**Security**:
- [Security Policy](security_policy.md)
- [RBAC Permission Model](rbac_permission_model.md)
- [Incident Response](incident_response.md)

### Developer

**Essential Reading**:
1. [API Documentation](api_documentation.md) - API overview
2. [API Reference](api/api_reference.md) - Endpoint details
3. [Architecture](architecture.md) - System design
4. [CLI Documentation](cli-documentation.md) - CLI tools

**Integration**:
- [Quick Start Guide](quickstart.md)
- [Advanced Embedding Models](advanced_embedding_models.md)
- [Distributed Services API](distributed_services_api.md)

**Testing**:
- [Test Execution Guide](TEST_EXECUTION_GUIDE.md)
- [Security Testing Framework](security_testing_framework.md)

### End User

**Essential Reading**:
1. [User Guide](UserGuide.md) - Complete user manual
2. [Quick Start Guide](quickstart.md) - Get started
3. [CLI Documentation](cli-documentation.md) - Command reference

**Operations**:
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)
- [API Documentation](api_documentation.md)

### DevOps Engineer

**Essential Reading**:
1. [Operations Runbook](operations_runbook.md) - Operational procedures
2. [Docker Deployment](DOCKER_DEPLOYMENT.md) - Container deployment
3. [Monitoring Setup](monitoring_setup.md) - Observability

**Advanced**:
- [Distributed Deployment](distributed_deployment_guide.md)
- [Performance Tuning](operations_runbook.md#performance-tuning)
- [Migration Guide](migration_guide_persistent_storage.md)

---

## Documentation Structure

```
docs/
├── README.md                              # This file
├── admin_endpoints.md                     # Admin API endpoints (NEW)
├── SHUTDOWN_FEATURE.md                    # Shutdown feature summary (NEW)
├── operations_runbook.md                  # Operations procedures (UPDATED)
│
├── Getting Started/
│   ├── quickstart.md                      # Quick start guide
│   ├── INSTALLATION_GUIDE.md              # Installation instructions
│   └── UserGuide.md                       # User guide
│
├── API Documentation/
│   ├── api_documentation.md               # API overview
│   ├── api/api_reference.md               # API reference (UPDATED)
│   ├── rbac_api_reference.md              # RBAC API
│   ├── persistence_api_reference.md       # Persistence API
│   └── distributed_services_api.md        # Distributed API
│
├── Administration/
│   ├── rbac_admin_guide.md                # RBAC administration
│   ├── CONFIGURATION_GUIDE.md             # Configuration
│   ├── TROUBLESHOOTING_GUIDE.md           # Troubleshooting
│   └── migration_guide_persistent_storage.md
│
├── Security/
│   ├── security_policy.md                 # Security policy
│   ├── rbac_permission_model.md           # Permissions
│   ├── zero_trust_architecture.md         # Security architecture
│   └── incident_response.md               # Incident response
│
├── Deployment/
│   ├── DOCKER_DEPLOYMENT.md               # Docker deployment
│   ├── LOCAL_DEPLOYMENT.md                # Local deployment
│   └── distributed_deployment_guide.md    # Distributed deployment
│
├── Advanced/
│   ├── architecture.md                    # Architecture
│   ├── vector_compression.md              # Compression
│   ├── gpu_acceleration.md                # GPU support
│   ├── advanced_embedding_models.md       # ML models
│   └── predictive_maintenance.md          # Maintenance
│
└── Testing/
    ├── TEST_EXECUTION_GUIDE.md            # Test guide
    └── security_testing_framework.md      # Security tests
```

---

## Getting Help

### Documentation Issues

If you find issues with the documentation:
1. Check the [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)
2. Review [Known Issues](KNOWN_ISSUES.md)
3. Open an issue on GitHub
4. Contact support

### Common Questions

**Q: How do I shut down the server?**
A: See [Shutdown Procedures](operations_runbook.md#shutdown-procedures) or [Admin Endpoints](admin_endpoints.md#shutdown-endpoint)

**Q: How do I manage users and roles?**
A: See [RBAC Admin Guide](rbac_admin_guide.md)

**Q: How do I configure the server?**
A: See [Configuration Guide](CONFIGURATION_GUIDE.md)

**Q: How do I deploy to production?**
A: See [Operations Runbook](operations_runbook.md) and [Docker Deployment](DOCKER_DEPLOYMENT.md)

**Q: Where can I find API examples?**
A: See [API Documentation](api_documentation.md) and [Quick Start Guide](quickstart.md)

---

## Contributing to Documentation

We welcome documentation improvements! When contributing:

1. Follow the existing structure and style
2. Include code examples where appropriate
3. Test all commands and code snippets
4. Update the table of contents
5. Add your changes to "Recent Updates" section
6. Submit a pull request

### Documentation Standards

- Use Markdown format
- Include clear headings and sections
- Provide working code examples
- Add diagrams where helpful
- Keep content up-to-date
- Link to related documents

---

## Support

- **Documentation**: You're here!
- **GitHub Issues**: https://github.com/jadevectordb/issues
- **Email**: support@jadevectordb.io
- **Community**: https://community.jadevectordb.io

---

## License

Documentation is licensed under CC BY 4.0
Code examples are licensed under MIT License
