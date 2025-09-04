import { describe, it, expect } from 'vitest';
import { BaseAdapter, EvaluationBatch } from '../adapter.js';
import { ComponentMap, DataInst, ReflectiveDataset, RolloutOutput, Trajectory } from '../../types/index.js';

// Test implementation of adapter
class TestAdapter extends BaseAdapter {
  async evaluate(
    batch: DataInst[],
    candidate: ComponentMap,
    captureTraces: boolean = false
  ): Promise<EvaluationBatch> {
    const outputs: RolloutOutput[] = [];
    const scores: number[] = [];
    const trajectories: Trajectory[] | null = captureTraces ? [] : null;
    
    for (const item of batch) {
      outputs.push({ result: 'test' });
      scores.push(1.0);
      
      if (trajectories) {
        trajectories.push({ input: item, output: 'test' });
      }
    }
    
    return { outputs, scores, trajectories };
  }
  
  async makeReflectiveDataset(
    candidate: ComponentMap,
    evalBatch: EvaluationBatch,
    componentsToUpdate: string[]
  ): Promise<ReflectiveDataset> {
    const dataset: ReflectiveDataset = {};
    
    for (const component of componentsToUpdate) {
      dataset[component] = [
        { input: 'test', output: 'test', feedback: 'Good' }
      ];
    }
    
    return dataset;
  }
}

describe('Adapter', () => {
  let adapter: TestAdapter;
  
  beforeEach(() => {
    adapter = new TestAdapter();
  });
  
  describe('evaluate', () => {
    it('should evaluate without traces', async () => {
      const batch = [{ data: 'test1' }, { data: 'test2' }];
      const candidate = { component1: 'text' };
      
      const result = await adapter.evaluate(batch, candidate, false);
      
      expect(result.outputs).toHaveLength(2);
      expect(result.scores).toHaveLength(2);
      expect(result.trajectories).toBeNull();
    });
    
    it('should evaluate with traces', async () => {
      const batch = [{ data: 'test1' }, { data: 'test2' }];
      const candidate = { component1: 'text' };
      
      const result = await adapter.evaluate(batch, candidate, true);
      
      expect(result.outputs).toHaveLength(2);
      expect(result.scores).toHaveLength(2);
      expect(result.trajectories).toHaveLength(2);
    });
  });
  
  describe('makeReflectiveDataset', () => {
    it('should create reflective dataset', async () => {
      const candidate = { component1: 'text', component2: 'text2' };
      const evalBatch: EvaluationBatch = {
        outputs: [{ result: 'out1' }],
        scores: [0.5],
        trajectories: [{ input: 'in', output: 'out1' }]
      };
      const componentsToUpdate = ['component1'];
      
      const dataset = await adapter.makeReflectiveDataset(
        candidate,
        evalBatch,
        componentsToUpdate
      );
      
      expect(dataset).toHaveProperty('component1');
      expect(dataset.component1).toBeInstanceOf(Array);
      expect(dataset.component1).toHaveLength(1);
    });
  });
});