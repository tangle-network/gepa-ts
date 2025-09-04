# Contributing to GEPA-TS

We welcome contributions to GEPA-TS! This guide will help you get started.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/gepa-ts.git
   cd gepa-ts
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Run tests**
   ```bash
   npm test
   ```

4. **Build the project**
   ```bash
   npm run build
   ```

## Development Workflow

### Code Style

- Use TypeScript strict mode
- Follow existing code patterns
- Use meaningful variable names
- Add JSDoc comments for public APIs

### Testing

All new features must include tests:

```typescript
describe('YourFeature', () => {
  it('should do something', async () => {
    // Test implementation
    expect(result).toBe(expected);
  });
});
```

For LLM-dependent code, use the record/replay infrastructure:

```typescript
import { TestLLM } from '../test/infrastructure/record-replay';

const testLLM = new TestLLM({ mode: 'replay' });
```

### Commit Messages

Use clear, descriptive commit messages:
- `feat: add new adapter for X`
- `fix: correct Pareto tracking bug`
- `docs: update API documentation`
- `test: add tests for merge proposer`
- `refactor: simplify state management`

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Ensure quality**
   ```bash
   npm run typecheck  # Type checking
   npm run lint       # Linting
   npm test          # Tests
   ```

4. **Submit PR**
   - Clear description of changes
   - Link related issues
   - Include test results

## Adding New Features

### New Adapter

1. Create adapter file in `src/adapters/`
2. Extend `BaseAdapter`
3. Implement required methods
4. Add tests in `test/adapters/`
5. Update exports in `src/adapters/index.ts`

Example:
```typescript
export class MyAdapter extends BaseAdapter<Input, Trace, Output> {
  async evaluate(batch, candidate, captureTraces) {
    // Implementation
  }
  
  async makeReflectiveDataset(candidate, evalBatch, components) {
    // Implementation
  }
}
```

### New Selection Strategy

1. Create strategy in `src/strategies/`
2. Implement `CandidateSelector` interface
3. Add tests
4. Export from index

### New Proposer

1. Create proposer in `src/proposer/`
2. Follow existing patterns (mutation, merge)
3. Include LLM interaction handling
4. Add comprehensive tests

## Architecture Guidelines

### Core Principles

1. **Type Safety**: Leverage TypeScript fully
2. **Immutability**: Prefer immutable operations
3. **Async First**: All I/O operations should be async
4. **Testability**: Design for testing

### Key Improvements to Maintain

The TypeScript implementation fixes critical bugs from Python:

1. **Per-Instance Pareto Tracking**: Keep the corrected implementation
   ```typescript
   // CORRECT - Track per instance, not globally
   paretoFrontValset: number[] = [];
   programAtParetoFrontValset: Set<number>[] = [];
   ```

2. **Proper Merge Algorithm**: Maintain ancestor-aware merging

### Performance Considerations

- Use efficient data structures (Set, Map)
- Minimize unnecessary copies
- Batch operations when possible
- Lazy evaluation for traces

## Documentation

### Code Documentation

Add JSDoc comments for:
- Public classes and methods
- Complex algorithms
- Type definitions

```typescript
/**
 * Evaluates candidates using tournament selection
 * @param state - Current optimization state
 * @param size - Tournament size
 * @returns Selected candidates
 */
selectCandidates(state: GEPAState, size: number): ComponentMap[]
```

### User Documentation

Update relevant docs when adding features:
- `README.md` - Usage examples
- `docs/API.md` - API reference
- `docs/MIGRATION.md` - Migration notes

## Testing Guidelines

### Test Coverage

Aim for >80% coverage:
```bash
npm run test:coverage
```

### Test Categories

1. **Unit Tests**: Individual components
2. **Integration Tests**: Component interactions
3. **End-to-End Tests**: Full optimization runs

### Deterministic Testing

Use record/replay for LLM interactions:
```bash
# Record real LLM responses
npm run test:record

# Replay for deterministic tests
npm run test:replay
```

## Release Process

1. Update version in `package.json`
2. Update CHANGELOG.md
3. Run full test suite
4. Build and verify
5. Tag release
6. Publish to npm

## Getting Help

- Open an issue for bugs
- Start a discussion for features
- Join our Discord (if applicable)
- Check existing issues first

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn
- Maintain professional discourse

## License

By contributing, you agree that your contributions will be licensed under the MIT License.