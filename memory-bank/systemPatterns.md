# StatTools System Patterns

## System Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Generators    │    │    Analysis     │    │    Filters      │
│                 │    │                 │    │                 │
│ • FBM Generator │    │ • DFA           │    │ • Kalman Filter │
│ • Kasdin Gen    │    │ • DPCCA         │    │ • Base Filter   │
│ • LBFBM Gen     │    │ • FA            │    │                 │
│ • Field Gen     │    │ • QSS           │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Utilities     │
                    │                 │
                    │ • Shared Buffer │
                    │ • Parallel Proc │
                    │ • Gamma Replace │
                    │ • Auxiliary     │
                    └─────────────────┘
```

### Module Organization
- **StatTools.generators**: Synthetic data generation with controlled statistical properties
- **StatTools.analysis**: Fluctuation analysis algorithms (DFA, DPCCA, FA, etc.)
- **StatTools.filters**: Signal processing and filtering operations
- **StatTools**: Core utilities and shared components

## Key Technical Decisions

### Parallel Processing Pattern
**Decision**: Use multi-threading with shared memory buffers for CPU-bound operations
**Rationale**:
- Large datasets require parallel processing for acceptable performance
- Python GIL limitations necessitate careful memory management
- Shared memory buffers enable efficient inter-thread communication
- Progress tracking requires thread-safe counters and locks

**Implementation**:
```python
# Shared buffer pattern for large matrix operations
buffer = SharedBuffer(shape=(n_vectors, vector_length), dtype=c_double)
# Thread-safe progress tracking
with tqdm(total=total_operations, desc="Processing") as pbar:
    # Parallel execution with progress updates
```

### Generator Interface Pattern
**Decision**: Unified interface for all data generators with configurable parameters
**Rationale**:
- Consistent API across different generation methods
- Easy extensibility for new generator types
- Parameter validation and error handling
- Progress tracking and cancellation support

**Implementation**:
```python
class BaseGenerator(ABC):
    def __init__(self, length: int, **kwargs):
        self.length = length
        # Common initialization

    @abstractmethod
    def generate(self, n_vectors: int, threads: int = 1) -> np.ndarray:
        pass
```

### Analysis Method Pattern
**Decision**: Class-based analysis methods with configuration objects
**Rationale**:
- Encapsulate analysis state and parameters
- Support for different analysis configurations
- Easy to extend with new analysis methods
- Consistent result format across methods

**Implementation**:
```python
class DFAAnalysis:
    def __init__(self, dataset: np.ndarray, degree: int = 2):
        self.dataset = dataset
        self.degree = degree

    def find_h(self) -> float:
        # Implementation
        return hurst_exponent
```

## Design Patterns

### Factory Pattern for Generators
**Context**: Multiple generator types with similar interfaces but different implementations
**Solution**: Factory functions that create appropriate generator instances
```python
def create_generator(generator_type: str, **params) -> BaseGenerator:
    if generator_type == "fbm":
        return FBMGenerator(**params)
    elif generator_type == "kasdin":
        return KasdinGenerator(**params)
    # etc.
```

### Strategy Pattern for Analysis Methods
**Context**: Different analysis algorithms that can be applied to the same data
**Solution**: Interchangeable analysis strategies with common interface
```python
class AnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, data: np.ndarray) -> AnalysisResult:
        pass

class DFAStrategy(AnalysisStrategy):
    def analyze(self, data: np.ndarray) -> AnalysisResult:
        # DFA implementation
        pass
```

### Observer Pattern for Progress Tracking
**Context**: Long-running operations need progress feedback
**Solution**: Progress observers that can be attached to operations
```python
class ProgressObserver:
    def update(self, progress: float, message: str):
        pass

class TQDMObserver(ProgressObserver):
    def __init__(self, total: int, desc: str):
        self.pbar = tqdm(total=total, desc=desc)

    def update(self, progress: float, message: str):
        self.pbar.update(1)
```

## Component Relationships

### Data Flow Patterns

#### Generation → Analysis Pipeline
```
Raw Parameters → Generator → Dataset → Analysis → Results
     ↓              ↓           ↓          ↓         ↓
Validation    Progress     Quality    Statistical  Visualization
              Tracking    Control    Validation    & Reporting
```

#### Parallel Processing Coordination
```
Main Thread → Worker Pool → Shared Memory → Result Aggregation
     ↓             ↓              ↓             ↓
Task Queue   Progress Sync   Buffer Mgmt    Final Results
Distribution   Updates       & Cleanup     & Validation
```

### Dependency Management

#### Core Dependencies
- **NumPy**: Fundamental array operations and mathematical functions
- **Multiprocessing**: Parallel execution and shared memory management
- **CTypes**: Interface between Python and C++ extensions

#### Optional Dependencies
- **SciPy**: Advanced mathematical functions (integration, optimization)
- **Matplotlib**: Visualization for research and debugging
- **Pandas**: Data manipulation for complex workflows

## Critical Implementation Paths

### Memory Management
**Pattern**: Pre-allocate large buffers and reuse them
**Rationale**: Avoid memory fragmentation during long-running processes
**Implementation**:
- SharedBuffer class for large array operations
- Memory pooling for frequently allocated objects
- Garbage collection hints for large computations

### Error Handling
**Pattern**: Graceful degradation with informative error messages
**Rationale**: Research code must be robust but informative when things go wrong
**Implementation**:
- Custom exception classes for different error types
- Parameter validation with descriptive messages
- Recovery mechanisms for common failure modes

### Performance Optimization
**Pattern**: Hybrid Python/C++ implementation
**Rationale**: Python for flexibility, C++ for performance-critical sections
**Implementation**:
- Python wrappers around C++ computational kernels
- Automatic fallback to pure Python for debugging
- Benchmarking framework for performance regression detection

## Communication Patterns

### Inter-Thread Communication
- **Shared Memory**: For large data structures that multiple threads need to access
- **Queues**: For task distribution and result collection
- **Locks**: For synchronizing access to shared resources
- **Events**: For coordination and cancellation signals

### External Interfaces
- **Command Line**: For batch processing and automation
- **Jupyter Integration**: For interactive research workflows
- **API**: RESTful interface for web-based applications (future)

## Quality Assurance Patterns

### Testing Strategy
- **Unit Tests**: Individual function and method validation
- **Integration Tests**: End-to-end workflow verification
- **Statistical Tests**: Validation against theoretical results
- **Performance Tests**: Regression detection and optimization

### Validation Patterns
- **Known Answer Tests**: Compare against published results
- **Statistical Property Tests**: Verify generated data properties
- **Edge Case Testing**: Robustness under extreme conditions
- **Cross-Platform Testing**: Consistency across different environments
