# JadeVectorDB Tutorials

Welcome to the JadeVectorDB tutorial collection! Choose your learning path based on your preferred interface and learning style.

## Available Tutorials

### 🌐 Web-Based Interactive Tutorial
**Path:** [`./web/`](./web/)

An immersive, browser-based learning environment with real-time visualizations and guided exercises.

**Best for:**
- Visual learners
- Those new to vector databases
- Understanding concepts through interactive exploration
- Quick prototyping and experimentation

**Features:**
- 📊 Real-time 2D/3D vector space visualizations
- 💻 Interactive code playground with syntax highlighting
- 🎯 6 progressive learning modules (Getting Started → Advanced Features)
- ✅ Progress tracking and achievements
- 🔄 Live API response preview
- 📝 Built-in assessment tools

**Start here:** [Web Tutorial README](./web/README.md)

---

### 💻 CLI-Based Tutorial
**Path:** [`./cli/`](./cli/)

Hands-on command-line exercises for practical, production-ready skills.

**Best for:**
- Developers and DevOps engineers
- Automation and scripting workflows
- Production deployment scenarios
- Those comfortable with terminal environments

**Features:**
- 📚 Comprehensive documentation (basics to advanced)
- 🏋️ 5 hands-on exercises with solutions
- 📊 Sample datasets for realistic scenarios
- 🔧 Scripts for common workflows
- ✅ Verification tools to check your work
- 🚀 Production best practices

**Start here:** [CLI Tutorial README](./cli/README.md)

---

## Quick Start Guide

### For Complete Beginners
1. Start with the **Web Tutorial** to understand vector database concepts
2. Complete modules 1-3 (Getting Started, Vector Manipulation, Advanced Search)
3. Switch to **CLI Tutorial** for practical, production-ready skills

### For Experienced Developers
1. Review the **CLI Tutorial basics** (15 minutes)
2. Jump into CLI hands-on exercises
3. Reference the **Web Tutorial** for visual understanding when needed

### For DevOps/Automation
1. Focus on **CLI Tutorial**
2. Complete all 5 exercises
3. Study the sample scripts for automation patterns

---

## Tutorial Comparison

| Feature | Web Tutorial | CLI Tutorial |
|---------|-------------|--------------|
| **Learning Style** | Interactive, visual | Hands-on, practical |
| **Time to Complete** | 2-3 hours | 3-4 hours |
| **Prerequisites** | Web browser | Terminal, CLI tools |
| **Environment** | Simulated API | Real JadeVectorDB instance |
| **Best For** | Concepts & exploration | Production skills |
| **Difficulty Curve** | Gentle | Moderate |
| **Visual Aids** | ✅ Extensive | ❌ None |
| **Automation Skills** | ❌ Limited | ✅ Comprehensive |
| **Code Examples** | JavaScript/Python | Shell/Python |

---

## Prerequisites

### Web Tutorial Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Node.js 18+ (for running locally)
- No JadeVectorDB instance required (uses simulation)

### CLI Tutorial Prerequisites
- JadeVectorDB running instance (typically `http://localhost:8080`)
- One of the following CLI tools:
  - Python CLI: `pip install -e cli/python`
  - Shell CLI: Available in `cli/shell/scripts/`
  - JavaScript CLI: Available in `cli/js/`
- Basic terminal/command-line knowledge
- API key for authentication

---

## Shared Resources

Both tutorials can leverage shared sample datasets:

### Sample Data Available
- **`shared/sample-data/products.json`** - E-commerce product embeddings
- **`shared/sample-data/documents.json`** - Document search vectors
- **`shared/sample-data/images.json`** - Image embedding examples

### Conceptual Documentation
- **`shared/concepts/vector-similarity.md`** - Understanding similarity metrics
- **`shared/concepts/indexing-strategies.md`** - Index types and performance
- **`shared/concepts/metadata-filtering.md`** - Advanced filtering techniques

---

## Learning Paths

### Path 1: Full Stack Developer
```
Web Tutorial (Modules 1-6) → CLI Tutorial (Exercises 1-3) → Production Deployment
```

### Path 2: Backend/API Developer
```
CLI Tutorial (All Exercises) → Web Tutorial (Modules 4-6 for advanced features)
```

### Path 3: Data Scientist
```
Web Tutorial (Modules 1-4) → CLI Tutorial (Exercise 2: Batch Operations) → Python SDK
```

### Path 4: DevOps Engineer
```
CLI Tutorial (All Exercises) → Deployment Documentation → Monitoring & Scaling
```

---

## Tutorial Structure

### Web Tutorial Modules
1. **Getting Started** - Introduction to vector databases and basic operations
2. **Vector Manipulation** - CRUD operations for vectors
3. **Advanced Search** - Similarity search techniques and parameters
4. **Metadata Filtering** - Complex queries with metadata conditions
5. **Index Management** - Understanding and configuring index types
6. **Advanced Features** - Embedding models, compression, and optimization

### CLI Tutorial Exercises
1. **Basics** - Environment setup, database creation, basic operations
2. **Batch Operations** - Efficient data import, error handling, progress monitoring
3. **Metadata Filtering** - Advanced search with complex filter conditions
4. **Index Management** - Performance optimization through index configuration
5. **Advanced Workflows** - Production patterns, monitoring, and maintenance

---

## Getting Help

### Documentation
- **API Documentation:** [`/docs/api_documentation.md`](../docs/api_documentation.md)
- **Architecture Guide:** [`/docs/architecture.md`](../docs/architecture.md)
- **CLI Documentation:** [`/cli/README.md`](../cli/README.md)

### Troubleshooting
- **Web Tutorial Issues:** Check browser console and [web/README.md](./web/README.md)
- **CLI Tutorial Issues:** See [cli/README.md](./cli/README.md) troubleshooting section
- **General Issues:** Refer to main [README.md](../README.md)

### Community
- **GitHub Issues:** Report bugs and feature requests
- **Contributing:** See [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Developer Guide:** See [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md)

---

## Next Steps After Tutorials

Once you've completed the tutorials:

1. **🚀 Production Deployment**
   - Review [DOCKER_DEPLOYMENT.md](../DOCKER_DEPLOYMENT.md)
   - Study Kubernetes configurations in `/k8s/`
   - Explore multi-cloud deployment in `/deployments/`

2. **📚 Advanced Topics**
   - GPU acceleration ([docs/gpu_acceleration.md](../docs/gpu_acceleration.md))
   - Vector compression ([docs/vector_compression.md](../docs/vector_compression.md))
   - Zero-trust architecture ([docs/zero_trust_architecture.md](../docs/zero_trust_architecture.md))

3. **🔧 Integration**
   - Integrate with your application using the Python/JS client libraries
   - Set up monitoring with Prometheus and Grafana
   - Implement backup and disaster recovery

4. **🎯 Real-World Projects**
   - Build a semantic search engine
   - Create a recommendation system
   - Implement image similarity search
   - Develop a RAG (Retrieval-Augmented Generation) application

---

## Tutorial Maintenance

These tutorials are actively maintained. If you find issues or have suggestions:

1. **Report Issues:** Open a GitHub issue with the `tutorial` label
2. **Suggest Improvements:** Submit a pull request
3. **Request Topics:** Open a discussion for new tutorial topics

---

**Happy Learning! 🎓**

Choose your path above and start exploring JadeVectorDB!
