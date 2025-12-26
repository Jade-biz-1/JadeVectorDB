# JadeVectorDB Installation Guide

## Table of Contents
- [Prerequisites](#prerequisites)
- [Backend Installation](#backend-installation)
- [Frontend Installation](#frontend-installation)
- [Default Users](#default-users)
- [Environment Configuration](#environment-configuration)
- [First Steps After Installation](#first-steps-after-installation)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+, Debian 11+), macOS 11+, or Windows WSL2
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Disk Space**: Minimum 2GB for installation, additional space for data storage

### Software Requirements

#### C++ Backend
- **Compiler**: C++20 compatible
  - GCC 11+ (recommended)
  - Clang 14+
  - MSVC 2022+ (Windows)
- **Build Tools**:
  - CMake 3.20 or higher
  - Make or Ninja
- **Dependencies** (automatically fetched by CMake):
  - Eigen (linear algebra)
  - FlatBuffers (serialization)
  - Apache Arrow (in-memory analytics)
  - gRPC (RPC framework)
  - Crow (HTTP framework)
  - Google Test (testing framework)

#### Frontend (Optional)
- **Node.js**: 16.x or higher
- **npm**: 8.x or higher (comes with Node.js)

## Backend Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Jade-biz-1/JadeVectorDB.git
cd JadeVectorDB
```

### Step 2: Build the Backend

```bash
cd backend
mkdir build
cd build
cmake ..
make -j$(nproc)  # Or use -j4 for 4 parallel jobs
```

**Build time**: Approximately 5-15 minutes depending on your system.

### Step 3: Verify Build

```bash
# Check if the main executable was created
ls -lh jadevectordb

# Run tests to verify installation
./jadevectordb_tests
```

### Step 4: Start the Server

```bash
# Start in development mode (default)
./jadevectordb

# Or explicitly set environment
export JADEVECTORDB_ENV=development
./jadevectordb
```

The server will start on **port 8080** by default.

**Verify server is running**:
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{"status": "healthy", "uptime_seconds": 5}
```

## Frontend Installation

### Step 1: Navigate to Frontend Directory

```bash
cd frontend
```

### Step 2: Install Dependencies

```bash
npm install
```

### Step 3: Start Development Server

```bash
npm run dev
```

The frontend will be available at **http://localhost:3000**

## Default Users

**IMPORTANT**: JadeVectorDB automatically creates default test users in development/test environments for your convenience.

### Default Credentials

| Username | Password | User ID | Roles | Use Case |
|----------|----------|---------|-------|----------|
| **admin** | admin123 | user_admin_default | admin, developer, user | Full administrative access, database management |
| **dev** | dev123 | user_dev_default | developer, user | Development and API testing |
| **test** | test123 | user_test_default | tester, user | Testing and QA workflows |

### When Are Default Users Created?

Default users are **automatically created** when you start the server in:
- **Development** mode (`JADEVECTORDB_ENV=development` or `JADEVECTORDB_ENV=dev`)
- **Test** mode (`JADEVECTORDB_ENV=test` or `JADEVECTORDB_ENV=testing`)
- **Local** mode (`JADEVECTORDB_ENV=local`)
- **Default** (when `JADEVECTORDB_ENV` is not set)

### When Are Default Users NOT Created?

Default users are **NOT created** when you start the server in:
- **Production** mode (`JADEVECTORDB_ENV=production` or `JADEVECTORDB_ENV=prod`)
- Any other environment value

**Security Note**: This is by design for security. Production deployments require you to create users with strong passwords manually.

### Environment Configuration

Control default user creation using the `JADEVECTORDB_ENV` environment variable:

```bash
# Development mode (creates default users) - DEFAULT
export JADEVECTORDB_ENV=development
./jadevectordb

# Test mode (creates default users)
export JADEVECTORDB_ENV=test
./jadevectordb

# Production mode (NO default users created)
export JADEVECTORDB_ENV=production
./jadevectordb
```

### Checking If Default Users Were Created

When the server starts, check the logs:

**Development/Test mode** - You should see:
```
[INFO] Seeding default users for development environment
[INFO] Created default user: admin with roles: [admin, developer, user]
[INFO] Created default user: dev with roles: [developer, user]
[INFO] Created default user: test with roles: [tester, user]
[INFO] Default user seeding complete: 3 created, 0 skipped
```

**Production mode** - You should see:
```
[INFO] Skipping default user seeding in production environment
```

**Subsequent starts** (users already exist):
```
[DEBUG] Default user 'admin' already exists, skipping
[DEBUG] Default user 'dev' already exists, skipping
[DEBUG] Default user 'test' already exists, skipping
[INFO] Default user seeding complete: 0 created, 3 skipped
```

## First Steps After Installation

### 1. Test API Authentication

```bash
# Login as admin
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

Expected response:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user_id": "user_admin_default",
  "username": "admin",
  "roles": ["admin", "developer", "user"]
}
```

**Save the token** - you'll need it for authenticated requests:
```bash
export TOKEN="<your-token-here>"
```

### 2. Create Your First Database

```bash
curl -X POST http://localhost:8080/v1/databases \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "name": "my_first_db",
    "dimension": 128,
    "metric": "cosine"
  }'
```

### 3. Access the Web UI

Open your browser and navigate to **http://localhost:3000**

Login using any of the default credentials:
- Username: `admin`
- Password: `admin123`

### 4. Explore the API

Full API documentation is available at:
- **API Docs**: `/docs/api_documentation.md`
- **Architecture**: `/docs/architecture.md`

## Troubleshooting

### Build Issues

**Issue**: CMake can't find required packages
```bash
# Solution: Update CMake and clear build cache
rm -rf build
mkdir build
cd build
cmake --version  # Verify CMake >= 3.20
cmake ..
```

**Issue**: Compilation errors related to C++20 features
```bash
# Solution: Check compiler version
gcc --version  # Should be >= 11.0
clang --version  # Should be >= 14.0

# Update if needed (Ubuntu/Debian):
sudo apt update
sudo apt install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
```

### Runtime Issues

**Issue**: Server won't start (port 8080 already in use)
```bash
# Solution: Check what's using port 8080
lsof -i :8080

# Kill the process or use a different port
export JDB_PORT=8081
./jadevectordb
```

**Issue**: Default users not created
```bash
# Solution: Check JADEVECTORDB_ENV variable
echo $JADEVECTORDB_ENV

# If set to production, change to development
export JADEVECTORDB_ENV=development
./jadevectordb

# Check logs to confirm user creation
```

**Issue**: Can't login with default credentials
```bash
# Verify the server is running in dev mode
curl http://localhost:8080/health

# Check server logs for user creation messages

# Try registering a new user manually
curl -X POST http://localhost:8080/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "myuser",
    "password": "MySecurePassword123!",
    "email": "myuser@example.com",
    "roles": ["user"]
  }'
```

### Frontend Issues

**Issue**: `npm install` fails
```bash
# Solution: Clear npm cache and retry
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

**Issue**: Frontend can't connect to backend
```bash
# Solution: Check API URL in frontend configuration
# Edit frontend/.env.local or frontend/src/lib/api.js
# Ensure API_BASE_URL points to http://localhost:8080
```

## Production Deployment

### Security Checklist

Before deploying to production:

- [ ] Set `JADEVECTORDB_ENV=production` to prevent default user creation
- [ ] Create admin users with strong passwords manually
- [ ] Enable SSL/TLS for HTTPS
- [ ] Configure firewall rules
- [ ] Set up proper logging and monitoring
- [ ] Change default ports if needed
- [ ] Review and configure authentication settings
- [ ] Enable backup and disaster recovery

### Creating Production Users

```bash
# Set production environment
export JADEVECTORDB_ENV=production
./jadevectordb &

# Create admin user with strong password
curl -X POST http://localhost:8080/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "prod_admin",
    "password": "YourVeryStrongPassword123!@#",
    "email": "admin@yourcompany.com",
    "roles": ["admin"]
  }'
```

## Next Steps

- Read the [User Guide](./UserGuide.md) for detailed usage instructions
- Explore the [API Documentation](./api_documentation.md)
- Review the [Architecture Overview](./architecture.md)
- Check out [Example Workflows](./examples/)
- Join the community discussions on GitHub

## Support

- **Documentation**: `/docs/`
- **GitHub Issues**: [Report bugs or request features](https://github.com/Jade-biz-1/JadeVectorDB/issues)
- **GitHub Discussions**: [Community discussions](https://github.com/Jade-biz-1/JadeVectorDB/discussions)

---

**Congratulations!** You've successfully installed JadeVectorDB. Happy vector searching! 🚀
