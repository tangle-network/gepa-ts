/**
 * Import Validation Tests - Exact 1:1 replica of Python GEPA's test_import.py
 * Ensures all GEPA-TS package imports work correctly
 */

import { describe, test } from 'vitest';
import { expect } from 'vitest';

describe('Package Import Tests', () => {
  test('test_package_import', () => {
    /**
     * Ensures the 'gepa-ts' package can be imported.
     * Exact replica of Python test_package_import function
     */
    try {
      // Import main GEPA module
      const gepa = require('../src/index.js');
      expect(gepa).toBeDefined();
    } catch (error) {
      expect.fail(`Failed to import the 'gepa-ts' package: ${error}`);
    }
  });

  test('test_gepa_optimize_import', () => {
    /**
     * Ensures the 'optimize' function can be imported.
     * Exact replica of Python test_gepa_optimize_import function
     */
    try {
      // Import optimize function specifically
      const { optimize } = require('../src/core/optimize.js');
      expect(optimize).toBeDefined();
      expect(typeof optimize).toBe('function');
    } catch (error) {
      expect.fail(`Failed to import the 'optimize' function: ${error}`);
    }
  });

  test('test_core_imports', () => {
    /**
     * Additional test to ensure all core components can be imported
     */
    try {
      const { GEPAOptimizer } = require('../src/core/optimizer.js');
      const { GEPAState } = require('../src/core/state.js');
      const { BaseAdapter } = require('../src/core/adapter.js');
      
      expect(GEPAOptimizer).toBeDefined();
      expect(GEPAState).toBeDefined();
      expect(BaseAdapter).toBeDefined();
    } catch (error) {
      expect.fail(`Failed to import core components: ${error}`);
    }
  });

  test('test_adapter_imports', () => {
    /**
     * Test that all adapters can be imported
     */
    try {
      const { DefaultAdapter } = require('../src/adapters/default-adapter.js');
      const { DSPyAdapter } = require('../src/adapters/dspy-adapter.js');
      const { AxLLMAdapter } = require('../src/adapters/axllm-adapter.js');
      const { MathAdapter } = require('../src/adapters/math-adapter.js');
      const { TerminalAdapter } = require('../src/adapters/terminal-adapter.js');
      
      expect(DefaultAdapter).toBeDefined();
      expect(DSPyAdapter).toBeDefined();  
      expect(AxLLMAdapter).toBeDefined();
      expect(MathAdapter).toBeDefined();
      expect(TerminalAdapter).toBeDefined();
    } catch (error) {
      expect.fail(`Failed to import adapters: ${error}`);
    }
  });

  test('test_model_imports', () => {
    /**
     * Test that language models can be imported
     */
    try {
      const { OpenAILanguageModel } = require('../src/models/openai.js');
      
      expect(OpenAILanguageModel).toBeDefined();
      expect(typeof OpenAILanguageModel).toBe('function'); // Constructor
    } catch (error) {
      expect.fail(`Failed to import language models: ${error}`);
    }
  });

  test('test_strategy_imports', () => {
    /**
     * Test that optimization strategies can be imported
     */
    try {
      const { TournamentSelector } = require('../src/strategies/tournament-selector.js');
      
      expect(TournamentSelector).toBeDefined();
    } catch (error) {
      expect.fail(`Failed to import strategies: ${error}`);
    }
  });

  test('test_proposer_imports', () => {
    /**
     * Test that proposers can be imported 
     */
    try {
      const { MergeProposer } = require('../src/proposer/merge-proposer.js');
      const { ReflectiveMutation } = require('../src/proposer/reflective-mutation.js');
      
      expect(MergeProposer).toBeDefined();
      expect(ReflectiveMutation).toBeDefined();
    } catch (error) {
      expect.fail(`Failed to import proposers: ${error}`);
    }
  });

  test('test_dataset_imports', () => {
    /**
     * Test that datasets can be imported
     */
    try {
      const { loadMathDataset } = require('../src/datasets/math.js');
      
      expect(loadMathDataset).toBeDefined();
      expect(typeof loadMathDataset).toBe('function');
    } catch (error) {
      expect.fail(`Failed to import datasets: ${error}`);
    }
  });

  test('test_example_imports', () => {
    /**
     * Test that examples can be imported and run
     */
    try {
      // These should be importable without errors
      const basicUsage = require('../examples/basic-usage.js');
      const trainAime = require('../examples/train-aime.js');
      
      expect(basicUsage).toBeDefined();
      expect(trainAime).toBeDefined();
    } catch (error) {
      expect.fail(`Failed to import examples: ${error}`);
    }
  });
});

describe('GEPA DSPy Dependency Test', () => {
  test('ensure_gepa_dspy_dependency', () => {
    /**
     * Exact replica of Python ensure_gepa_dspy_dependency.py
     * Ensures DSPy integration works correctly
     */
    try {
      // Import DSPy-related components
      const { DSPyAdapter } = require('../src/adapters/dspy-adapter.js');
      const { DSPyFullProgramAdapter } = require('../src/adapters/dspy-full-program-adapter.js');
      
      expect(DSPyAdapter).toBeDefined();
      expect(DSPyFullProgramAdapter).toBeDefined();
      
      // Ensure they can be instantiated (basic check)
      const mockConfig = {
        studentModule: {
          modules: [],
          flow: 'test'
        },
        metricFn: () => 1.0,
        lm: () => Promise.resolve('test response')
      };
      
      const adapter = new DSPyAdapter(mockConfig);
      expect(adapter).toBeInstanceOf(DSPyAdapter);
      
    } catch (error) {
      expect.fail(`GEPA DSPy dependency check failed: ${error}`);
    }
  });
});