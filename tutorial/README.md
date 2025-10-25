# JadeVectorDB Interactive Tutorial

An immersive, browser-based playground environment that allows users to learn JadeVectorDB through hands-on experience with guided tutorials, real-time visualizations, and instant feedback mechanisms.

## Overview

The Interactive Tutorial provides a comprehensive learning experience that combines theoretical knowledge with practical, hands-on practice in a simulated environment. Users can explore all major JadeVectorDB features without needing to set up a real database or risk affecting production systems.

## Features

- **Interactive Playground**: Real-time vector space visualization with t-SNE/UMAP projections
- **Progressive Learning Modules**: Guided tutorials from basic concepts to advanced features
- **Real-time Visualizations**: 2D/3D interactive plots showing vector positions and relationships
- **Code Editor**: Syntax-highlighted editor with auto-completion for API calls
- **Live Preview**: Instant results panel showing search outcomes, vector positions, and metrics
- **Assessment Tools**: Interactive quizzes and knowledge checks
- **Achievement System**: Badges for completing tasks and mastering concepts

## Architecture

The tutorial is built as a standalone Next.js application that simulates the JadeVectorDB API responses. This allows users to learn the system without needing a live database connection.

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Clone the repository
2. Navigate to the tutorial directory: `cd tutorial`
3. Install dependencies: `npm install`
4. Start the development server: `npm run dev`

The tutorial will be accessible at `http://localhost:3000`

### Development

During development, the tutorial can be run with hot reloading:

```bash
npm run dev
```

### Building for Production

To create an optimized production build:

```bash
npm run build
```

## Project Structure

```
tutorial/
├── src/
│   ├── components/      # React components for UI elements
│   ├── modules/         # Tutorial modules (1-6)
│   ├── contexts/        # React context providers
│   ├── lib/             # Utility functions and helpers
│   ├── scenarios/       # Real-world use case implementations
│   └── __tests__/       # Test files
├── public/              # Static assets
├── package.json         # Project dependencies and scripts
└── README.md            # This file
```

## Contributing

We welcome contributions to improve the interactive tutorial. Please see our CONTRIBUTING.md file for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.