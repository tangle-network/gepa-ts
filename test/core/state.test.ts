/**
 * Tests for GEPAState with per-instance Pareto front tracking
 * This tests the critical fix where Pareto fronts are tracked per validation instance
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import { GEPAState } from '../../src/core/state.js';
import { ComponentMap } from '../../src/types/index.js';

describe('GEPAState', () => {
  let state: GEPAState;
  
  beforeEach(() => {
    state = new GEPAState();
  });
  
  describe('Per-Instance Pareto Front Tracking', () => {
    it('should track Pareto fronts independently for each validation instance', () => {
      // This is the critical test for the bug fix
      // Previously, Pareto was tracked globally which was WRONG
      
      const numInstances = 3;
      const numPrograms = 4;
      
      // Initialize with validation instances
      state.initialize(numInstances);
      
      // Add programs with different scores per instance
      const programs: ComponentMap[] = [];
      const scoresByInstance: number[][] = [
        [0.5, 0.7, 0.3, 0.9], // Instance 0 scores for each program
        [0.8, 0.2, 0.6, 0.4], // Instance 1 scores for each program  
        [0.3, 0.5, 0.8, 0.7], // Instance 2 scores for each program
      ];
      
      // Add programs
      for (let p = 0; p < numPrograms; p++) {
        const program = { text: `Program ${p}` };
        programs.push(program);
        state.addProgram(program, {});
      }
      
      // Update scores for each instance
      for (let instance = 0; instance < numInstances; instance++) {
        for (let prog = 0; prog < numPrograms; prog++) {
          state.updateParetoFront(instance, prog, scoresByInstance[instance][prog]);
        }
      }
      
      // Verify Pareto fronts per instance
      expect(state.paretoFrontValset.length).toBe(numInstances);
      expect(state.programAtParetoFrontValset.length).toBe(numInstances);
      
      // Check that each instance has its own best score
      expect(state.paretoFrontValset[0]).toBe(0.9); // Best for instance 0
      expect(state.paretoFrontValset[1]).toBe(0.8); // Best for instance 1
      expect(state.paretoFrontValset[2]).toBe(0.8); // Best for instance 2
      
      // Check that correct programs are on Pareto front
      expect(state.programAtParetoFrontValset[0].has(3)).toBe(true); // Program 3 best for instance 0
      expect(state.programAtParetoFrontValset[1].has(0)).toBe(true); // Program 0 best for instance 1
      expect(state.programAtParetoFrontValset[2].has(2)).toBe(true); // Program 2 best for instance 2
    });
    
    it('should update Pareto front when better program is found', () => {
      state.initialize(2);
      
      const prog1 = { text: 'Program 1' };
      const prog2 = { text: 'Program 2' };
      
      state.addProgram(prog1, {});
      state.addProgram(prog2, {});
      
      // Initial scores
      state.updateParetoFront(0, 0, 0.5);
      state.updateParetoFront(0, 1, 0.3);
      
      expect(state.paretoFrontValset[0]).toBe(0.5);
      expect(state.programAtParetoFrontValset[0].has(0)).toBe(true);
      
      // Better program for instance 0
      state.updateParetoFront(0, 1, 0.7);
      
      expect(state.paretoFrontValset[0]).toBe(0.7);
      expect(state.programAtParetoFrontValset[0].has(1)).toBe(true);
      expect(state.programAtParetoFrontValset[0].has(0)).toBe(false);
    });
    
    it('should handle multiple programs at Pareto front', () => {
      state.initialize(1);
      
      const prog1 = { text: 'Program 1' };
      const prog2 = { text: 'Program 2' };
      const prog3 = { text: 'Program 3' };
      
      state.addProgram(prog1, {});
      state.addProgram(prog2, {});
      state.addProgram(prog3, {});
      
      // All programs have same score for instance 0
      state.updateParetoFront(0, 0, 0.8);
      state.updateParetoFront(0, 1, 0.8);
      state.updateParetoFront(0, 2, 0.8);
      
      // All should be on Pareto front
      expect(state.paretoFrontValset[0]).toBe(0.8);
      expect(state.programAtParetoFrontValset[0].size).toBe(3);
      expect(state.programAtParetoFrontValset[0].has(0)).toBe(true);
      expect(state.programAtParetoFrontValset[0].has(1)).toBe(true);
      expect(state.programAtParetoFrontValset[0].has(2)).toBe(true);
    });
  });
  
  describe('Program Management', () => {
    it('should add and retrieve programs', () => {
      const prog1: ComponentMap = { 
        instruction: 'Do task A',
        format: 'Output JSON'
      };
      const metadata1 = { generation: 1, parent: null };
      
      const prog2: ComponentMap = {
        instruction: 'Do task B',
        format: 'Output XML'
      };
      const metadata2 = { generation: 2, parent: 0 };
      
      const idx1 = state.addProgram(prog1, metadata1);
      const idx2 = state.addProgram(prog2, metadata2);
      
      expect(idx1).toBe(0);
      expect(idx2).toBe(1);
      
      expect(state.getProgram(idx1)).toEqual(prog1);
      expect(state.getProgram(idx2)).toEqual(prog2);
      
      expect(state.getProgramMetadata(idx1)).toEqual(metadata1);
      expect(state.getProgramMetadata(idx2)).toEqual(metadata2);
      
      expect(state.getAllPrograms()).toEqual([prog1, prog2]);
      expect(state.getProgramCount()).toBe(2);
    });
    
    it('should track program existence', () => {
      const prog = { text: 'Test program' };
      
      expect(state.hasProgram(prog)).toBe(false);
      
      state.addProgram(prog, {});
      
      expect(state.hasProgram(prog)).toBe(true);
      expect(state.hasProgram({ text: 'Test program' })).toBe(true); // Same content
      expect(state.hasProgram({ text: 'Different' })).toBe(false);
    });
    
    it('should not add duplicate programs', () => {
      const prog = { text: 'Duplicate test' };
      
      const idx1 = state.addProgram(prog, { gen: 1 });
      const idx2 = state.addProgram(prog, { gen: 2 });
      
      expect(idx1).toBe(0);
      expect(idx2).toBe(0); // Returns existing index
      expect(state.getProgramCount()).toBe(1);
      
      // Metadata should not be updated
      expect(state.getProgramMetadata(0)).toEqual({ gen: 1 });
    });
  });
  
  describe('Score Management', () => {
    it('should track scores for train and validation sets', () => {
      state.initialize(2);
      
      const prog = { text: 'Test' };
      state.addProgram(prog, {});
      
      // Set train scores
      state.setTrainScores(0, [0.6, 0.7, 0.8]);
      expect(state.getTrainScores(0)).toEqual([0.6, 0.7, 0.8]);
      expect(state.getAverageTrainScore(0)).toBeCloseTo(0.7);
      
      // Set validation scores
      state.setValScores(0, [0.5, 0.9]);
      expect(state.getValScores(0)).toEqual([0.5, 0.9]);
      expect(state.getAverageValScore(0)).toBeCloseTo(0.7);
    });
    
    it('should handle missing scores gracefully', () => {
      state.initialize(1);
      
      const prog = { text: 'Test' };
      state.addProgram(prog, {});
      
      expect(state.getTrainScores(0)).toBeUndefined();
      expect(state.getValScores(0)).toBeUndefined();
      expect(state.getAverageTrainScore(0)).toBe(0);
      expect(state.getAverageValScore(0)).toBe(0);
    });
  });
  
  describe('Best Program Selection', () => {
    it('should track best program by validation score', () => {
      state.initialize(3);
      
      const prog1 = { text: 'Program 1' };
      const prog2 = { text: 'Program 2' };
      const prog3 = { text: 'Program 3' };
      
      state.addProgram(prog1, {});
      state.addProgram(prog2, {});
      state.addProgram(prog3, {});
      
      state.setValScores(0, [0.5, 0.6, 0.4]);
      state.setValScores(1, [0.7, 0.8, 0.9]);
      state.setValScores(2, [0.3, 0.2, 0.4]);
      
      // Update Pareto fronts
      for (let inst = 0; inst < 3; inst++) {
        for (let prog = 0; prog < 3; prog++) {
          const scores = state.getValScores(prog)!;
          state.updateParetoFront(inst, prog, scores[inst]);
        }
      }
      
      const best = state.getBestProgram();
      expect(best).not.toBeNull();
      expect(best!.program).toEqual(prog2);
      expect(best!.score).toBeCloseTo(0.633); // Average of [0.7, 0.8, 0.9]
    });
    
    it('should return null when no programs exist', () => {
      state.initialize(1);
      expect(state.getBestProgram()).toBeNull();
    });
  });
  
  describe('Statistics', () => {
    it('should provide accurate statistics', () => {
      state.initialize(2);
      
      const prog1 = { text: 'P1' };
      const prog2 = { text: 'P2' };
      
      state.addProgram(prog1, { generation: 1 });
      state.addProgram(prog2, { generation: 2 });
      
      state.setTrainScores(0, [0.6, 0.7]);
      state.setTrainScores(1, [0.8, 0.9]);
      state.setValScores(0, [0.5, 0.6]);
      state.setValScores(1, [0.7, 0.8]);
      
      const stats = state.getStatistics();
      
      expect(stats.totalPrograms).toBe(2);
      expect(stats.uniquePrograms).toBe(2);
      expect(stats.averageTrainScore).toBeCloseTo(0.75);
      expect(stats.averageValScore).toBeCloseTo(0.65);
      expect(stats.bestTrainScore).toBe(0.85);
      expect(stats.bestValScore).toBe(0.75);
      
      // Pareto front info
      for (let i = 0; i < 2; i++) {
        state.updateParetoFront(i, 0, state.getValScores(0)![i]);
        state.updateParetoFront(i, 1, state.getValScores(1)![i]);
      }
      
      const statsWithPareto = state.getStatistics();
      expect(statsWithPareto.paretoFrontSizes).toHaveLength(2);
    });
  });
  
  describe('State Serialization', () => {
    it('should export and import state correctly', () => {
      state.initialize(2);
      
      const prog1 = { text: 'Program 1', version: '1.0' };
      const prog2 = { text: 'Program 2', version: '2.0' };
      
      state.addProgram(prog1, { gen: 1 });
      state.addProgram(prog2, { gen: 2 });
      
      state.setTrainScores(0, [0.5, 0.6]);
      state.setValScores(0, [0.7, 0.8]);
      
      // Export
      const exported = state.exportState();
      
      // Create new state and import
      const newState = new GEPAState();
      newState.importState(exported);
      
      // Verify
      expect(newState.getProgramCount()).toBe(2);
      expect(newState.getProgram(0)).toEqual(prog1);
      expect(newState.getProgram(1)).toEqual(prog2);
      expect(newState.getTrainScores(0)).toEqual([0.5, 0.6]);
      expect(newState.getValScores(0)).toEqual([0.7, 0.8]);
      expect(newState.paretoFrontValset).toEqual(state.paretoFrontValset);
    });
    
    it('should handle empty state export/import', () => {
      const exported = state.exportState();
      
      const newState = new GEPAState();
      newState.importState(exported);
      
      expect(newState.getProgramCount()).toBe(0);
      expect(newState.paretoFrontValset).toEqual([]);
    });
  });
  
  describe('Edge Cases', () => {
    it('should handle very large number of programs', () => {
      state.initialize(1);
      
      const numPrograms = 1000;
      for (let i = 0; i < numPrograms; i++) {
        state.addProgram({ id: `prog-${i}` }, {});
      }
      
      expect(state.getProgramCount()).toBe(numPrograms);
      
      // Update Pareto with random scores
      for (let i = 0; i < numPrograms; i++) {
        state.updateParetoFront(0, i, Math.random());
      }
      
      expect(state.paretoFrontValset[0]).toBeGreaterThan(0);
      expect(state.programAtParetoFrontValset[0].size).toBeGreaterThan(0);
    });
    
    it('should handle programs with complex nested structures', () => {
      const complexProgram: ComponentMap = {
        instruction: 'Main instruction',
        subComponents: {
          nested: {
            deeply: {
              value: 'Deep value'
            }
          },
          array: [1, 2, 3],
          mixed: [{ a: 1 }, { b: 2 }]
        },
        metadata: {
          timestamp: Date.now(),
          version: '1.0.0'
        }
      };
      
      const idx = state.addProgram(complexProgram, { complex: true });
      const retrieved = state.getProgram(idx);
      
      expect(retrieved).toEqual(complexProgram);
      expect(state.hasProgram(complexProgram)).toBe(true);
    });
  });
});