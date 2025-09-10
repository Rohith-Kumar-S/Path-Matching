# Path-Matching
Multi-Source Multi-Target Pathfinding with Bipartite Matching

An interactive visualization tool for solving optimal assignment problems using shortest path algorithms and bipartite matching. This project demonstrates the integration of pathfinding algorithms with assignment optimization, applicable to real-world scenarios like ride-sharing, delivery routing, and emergency vehicle dispatch.

## Features

### Core Capabilities
- **Graph Generation**
  - When the user runs the application, a graph is generated.
  - The graph is based on 8-directional grids from the selected obstacle map.
  - Each node/block in the grid is connected to its neighboring nodes.
  - Neighboring nodes are connected only if they are not obstacles.
  
- **Multiple Pathfinding Algorithms**
  - Dijkstra's Algorithm
  - A* Search (with multiple heuristics)
  - Breadth-First Search (BFS)

- **Bipartite Matching Methods**
  - Hungarian Algorithm (Linear Programming via SciPy)
  - Greedy Algorithm (nearest-available assignment)

- **Interactive Grid Interface**
  - Customizable obstacle maps
  - Click-to-place sources and targets
  - Support for multiple source-target combinations
  - Real-time path visualization

- **Performance Metrics**
  - Algorithm execution time comparison
  - Path distance calculations
  - Matching quality metrics

## Getting Started

### Prerequisites
```bash
Python 3.8+
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pathfinding-bipartite-matching.git
cd pathfinding-bipartite-matching
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `streamlit` - Web application framework
- `numpy` - Numerical computations
- `scipy` - Linear programming for optimal matching
- `matplotlib` - Visualization support

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 How to Use

### 1. **Map Setup**
   - Select a pre-defined obstacle map or create custom obstacles
   - Black cells represent impassable obstacles

### 2. **Place Sources and Targets**
   - Click on the grid to place sources (displayed in **red**)
   - Click to place targets (displayed in **blue**)
   - Support for multiple sources and targets

### 3. **Algorithm Selection**
   - **Pathfinding Algorithm**: Choose between Dijkstra, A*, or BFS
   - **For A***: Select heuristic (Manhattan, Euclidean, or Diagonal)
   - **Matching Method**: Choose Greedy or Linear Programming (for multiple sources/targets)

### 4. **Execute and Visualize**
   - Click "Find Paths" to run the algorithms
   - **Yellow cells**: Final optimal path from source to target
   - **Other colors**: Search space explored by the algorithm
   - View metrics dashboard for performance comparison

## Visualization Guide

| Color | Meaning |
|-------|---------|
| Red | Source points |
| Blue | Target points |
| Black | Obstacles |
| Yellow | Final shortest path |
| Other Colors | Algorithm exploration paths (one color per source) |

## Algorithm Details

### Pathfinding Algorithms

#### **Dijkstra's Algorithm**
- Guarantees shortest path in weighted graphs
- Explores nodes in order of distance from source
- Time Complexity: O((V + E) log V)

#### **A* Search**
- Uses heuristic to guide search toward target
- Faster than Dijkstra for single target
- Heuristic options:
  - **Manhattan**: |x₁-x₂| + |y₁-y₂| 
  - **Euclidean**: √((x₁-x₂)² + (y₁-y₂)²)

#### **Breadth-First Search (BFS)**
- Optimal for unweighted graphs
- Explores level by level
- Time Complexity: O(V + E)

### Bipartite Matching

#### **Linear Programming**
- Finds globally optimal assignment
- Minimizes total distance across all assignments
- Time Complexity: O(n³)

#### **Greedy Algorithm**
- Assigns each source to nearest available target
- Faster but potentially suboptimal
- Time Complexity: O(n² log n)

## Applications

This project demonstrates solutions applicable to:

- **Ride-Sharing Services**: Optimal driver-passenger matching
- **Delivery Logistics**: Driver-to-order assignment
- **Emergency Services**: Ambulance-to-hospital routing
- **Warehouse Automation**: Robot-to-task allocation
- **Game AI**: Multi-unit pathfinding and target assignment
- **Network Routing**: Optimal resource allocation

## Project Structure

```
project/
├── app.py                 # Main Streamlit application
├── utils/
│   ├── algorithms_impl.py # Algorithm implementations
│   └── graph_gen.py       # Generates graph from a given obstacle map
|   └── simple_graph_modified.py # Graph class
└── README.md
```

## Performance Metrics

The application tracks and displays:
- **Pathfinding Time**: Time to compute all shortest paths
- **Matching Time**: Time to solve assignment problem
- **Total Distance**: Sum of all assigned path lengths
- **Optimality Gap**: Difference between greedy and optimal solutions
- **Search Space**: Nodes explored during pathfinding

## Technical Implementation

- **Language**: Python 3.8+
- **Web Framework**: Streamlit
- **Pathfinding**: Custom implementations from scratch
- **Optimization**: SciPy's `linprog` for weighted bipartite matching
- **Visualization**: Numpy and OpenCv integration with Streamlit
- **Data Structures**: NumPy arrays for grid representation

## Future Enhancements

- Dynamic obstacles and moving targets
- Additional algorithms (D*, Jump Point Search)
- Path smoothing and post-processing
- Export results to CSV/JSON
