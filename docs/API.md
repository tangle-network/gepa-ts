# API Documentation

## Core Classes

### GEPAOptimizer

The main optimization orchestrator.

```typescript
class GEPAOptimizer {
  constructor(options: {
    adapter: BaseAdapter;
    config?: OptimizationConfig;
    wandbIntegration?: WandBIntegration;
  })
  
  async optimize(params: OptimizationParams): Promise<OptimizationResult>
}
```

#### OptimizationConfig

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| populationSize | number | 10 | Number of candidates per generation |
| generations | number | 10 | Maximum generations to run |
| mutationRate | number | 0.3 | Probability of mutation (0-1) |
| mergeRate | number | 0.2 | Probability of merging (0-1) |
| eliteRatio | number | 0.1 | Fraction of population preserved |
| tournamentSize | number | 3 | Tournament selection size |
| verbose | boolean | false | Enable detailed logging |
| runDir | string? | null | Directory for state persistence |
| seed | number? | null | Random seed for reproducibility |

#### OptimizationParams

```typescript
interface OptimizationParams {
  initialCandidate: ComponentMap;      // Starting text components
  trainData: DataInst[];               // Training examples
  valData?: DataInst[];                // Validation examples
  componentsToUpdate?: string[];       // Components to optimize
  seedPopulation?: ComponentMap[];     // Additional candidates
}
```

#### OptimizationResult

```typescript
interface OptimizationResult {
  bestCandidate: ComponentMap;         // Best performing candidate
  bestScore: number;                    // Best validation score
  generationsCompleted: number;         // Iterations completed
  totalEvaluations: number;            // Total LLM calls
  history: GenerationHistory[];         // Per-generation metrics
}
```

## Adapters

### BaseAdapter

Abstract base class for all adapters.

```typescript
abstract class BaseAdapter<D, T, O> {
  abstract async evaluate(
    batch: D[],
    candidate: ComponentMap,
    captureTraces: boolean
  ): Promise<EvaluationBatch<T, O>>;
  
  abstract async makeReflectiveDataset(
    candidate: ComponentMap,
    evalBatch: EvaluationBatch<T, O>,
    componentsToUpdate: string[]
  ): Promise<ReflectiveDataset>;
  
  // Optional: custom text proposal
  proposeNewTexts?(
    candidate: ComponentMap,
    reflectiveDataset: ReflectiveDataset,
    componentsToUpdate: string[]
  ): ComponentMap;
}
```

### DefaultAdapter

Simple single-turn evaluation adapter.

```typescript
new DefaultAdapter({
  evaluator: async (prompt: string, example: DataInst) => number,
  batchSize?: number,
  traceCapture?: async (prompt: string, example: DataInst) => any
})
```

### DSPyAdapter

DSPy framework integration for structured prompting.

```typescript
new DSPyAdapter({
  lm: async (prompt: string) => string,
  modules: {
    [name: string]: {
      type: 'Predict' | 'ChainOfThought' | 'ReAct',
      signature: {
        inputs: string[],
        outputs: string[],
        instruction: string
      }
    }
  }
})
```

### AnyMathsAdapter

Mathematical problem-solving specialization.

```typescript
new AnyMathsAdapter({
  model: async (prompt: string) => string | ((prompt: string) => string),
  extractAnswer?: (response: string) => string | number,
  checkAnswer?: (predicted: any, expected: any) => boolean,
  requireSteps?: boolean,
  timeout?: number
})
```

### TerminalBenchAdapter

Terminal agent optimization.

```typescript
new TerminalBenchAdapter({
  agentExecutor: async (
    instruction: string,
    task: string,
    initialState: any
  ) => TerminalTrace,
  maxCommands?: number,
  timeout?: number,
  verbose?: boolean
})
```

## Selection Strategies

### CandidateSelector

```typescript
interface CandidateSelector {
  selectCandidates(
    state: GEPAState,
    componentsToUpdate: string[],
    numCandidates?: number
  ): ComponentMap[];
}
```

### Available Selectors

#### ParetoCandidateSelector
Selects from Pareto-optimal candidates.

```typescript
new ParetoCandidateSelector()
```

#### CurrentBestCandidateSelector
Always selects the highest-scoring candidate.

```typescript
new CurrentBestCandidateSelector()
```

#### TournamentCandidateSelector
Tournament-based selection.

```typescript
new TournamentCandidateSelector({
  tournamentSize: number,
  selectionPressure?: number,  // 0-1, default 0.8
  replacement?: boolean,        // default false
  fitnessFunction?: 'validation' | 'training' | 'combined'
})
```

#### RandomCandidateSelector
Random selection.

```typescript
new RandomCandidateSelector()
```

## Proposers

### MutationProposer

LLM-based text mutation.

```typescript
new MutationProposer(
  lm: async (prompt: string) => string
)

async propose(
  candidate: ComponentMap,
  reflectiveDataset: ReflectiveDataset,
  componentsToUpdate: string[]
): Promise<ComponentMap>
```

### MergeProposer

Ancestor-aware candidate merging.

```typescript
new MergeProposer(
  lm: async (prompt: string) => string
)

async propose(
  candidate1: ComponentMap,
  candidate2: ComponentMap,
  componentsToUpdate: string[],
  commonAncestor?: ComponentMap
): Promise<ComponentMap>
```

### DSPyInstructionProposer

DSPy-specific instruction optimization.

```typescript
new DSPyInstructionProposer(
  lm: async (prompt: string) => string
)

async proposeInstruction(
  currentInstruction: string,
  datasetWithFeedback: Array<Record<string, any>>
): Promise<string>

async proposeVariants(
  currentInstruction: string,
  datasetWithFeedback: Array<Record<string, any>>,
  numVariants?: number
): Promise<string[]>
```

## State Management

### GEPAState

Manages optimization state and Pareto fronts.

```typescript
class GEPAState {
  // Initialize with validation instances
  initialize(numValidationInstances: number): void
  
  // Program management
  addProgram(program: ComponentMap, metadata: any): number
  getProgram(index: number): ComponentMap
  hasProgram(program: ComponentMap): boolean
  
  // Score tracking
  setTrainScores(programIndex: number, scores: number[]): void
  setValScores(programIndex: number, scores: number[]): void
  
  // Pareto front (per-instance)
  updateParetoFront(
    instanceIndex: number,
    programIndex: number,
    score: number
  ): void
  
  // Best program selection
  getBestProgram(): { program: ComponentMap, score: number } | null
  
  // Serialization
  exportState(): any
  importState(data: any): void
}
```

## Integrations

### WandBIntegration

Weights & Biases experiment tracking.

```typescript
new WandBIntegration({
  project: string,
  entity?: string,
  name?: string,
  tags?: string[],
  config?: Record<string, any>,
  mode?: 'online' | 'offline' | 'disabled'
})

// Methods
async init(): Promise<void>
async logGeneration(
  generation: number,
  state: GEPAState,
  evalResults?: any
): Promise<void>
async logComponentUpdate(
  generation: number,
  componentName: string,
  oldText: string,
  newText: string,
  improvement: number
): Promise<void>
async finish(): Promise<void>
```

## Testing Utilities

### TestLLM

Deterministic LLM for testing.

```typescript
new TestLLM({
  mode?: 'record' | 'replay' | 'passthrough',
  sessionId?: string,
  realLLM?: (prompt: string) => Promise<string>
})

async generate(prompt: string): Promise<string>
finish(): void
```

### RecordReplayManager

Record and replay LLM calls.

```typescript
new RecordReplayManager({
  mode?: 'record' | 'replay' | 'passthrough',
  recordingPath?: string,
  strict?: boolean
})

async execute(
  input: string,
  actualExecutor: (input: string) => Promise<string>,
  metadata?: any
): Promise<string>
```

## Type Definitions

### ComponentMap
```typescript
type ComponentMap = Record<string, string>
```

### DataInst
```typescript
interface DataInst {
  [key: string]: any  // Domain-specific data
}
```

### EvaluationBatch
```typescript
interface EvaluationBatch<T, O> {
  outputs: O[];
  scores: number[];
  trajectories: T[] | null;
}
```

### ReflectiveDataset
```typescript
type ReflectiveDataset = Record<string, Array<Record<string, any>>>
```