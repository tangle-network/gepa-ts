import { describe, it, expect, beforeEach } from 'vitest';
import { GEPAState } from '../state.js';
import { ComponentMap, RolloutOutput } from '../../types/index.js';

describe('GEPAState', () => {
  let state: GEPAState;
  
  beforeEach(() => {
    state = new GEPAState();
  });
  
  describe('initialization', () => {
    it('should initialize with default values', () => {
      expect(state.i).toBe(0);
      expect(state.totalNumEvals).toBe(0);
      expect(state.numFullDsEvals).toBe(0);
      expect(state.programs).toEqual([]);
      expect(state.programParents).toEqual([]);
      expect(state.programFullScoresValSet).toEqual([]);
      expect(state.programSubScoresValSet).toEqual([]);
      expect(state.paretoFrontValset).toEqual([]);
      expect(state.programAtParetoFrontValset).toEqual([]);
      expect(state.fullProgramTrace).toEqual([]);
    });
    
    it('should handle trackBestOutputs flag', () => {
      const stateWithTracking = new GEPAState(true);
      expect(stateWithTracking.trackBestOutputs).toBe(true);
      
      const stateWithoutTracking = new GEPAState(false);
      expect(stateWithoutTracking.trackBestOutputs).toBe(false);
    });
  });
  
  describe('addProgram', () => {
    it('should add a program to the state', () => {
      const program: ComponentMap = { system_prompt: 'Test prompt' };
      const outputs: RolloutOutput[] = [{ result: 'output1' }, { result: 'output2' }];
      const subscores = [0.8, 0.9];
      const avgScore = 0.85;
      
      const [idx, paretoIdx] = state.addProgram(
        program,
        [],
        avgScore,
        outputs,
        subscores,
        0
      );
      
      expect(idx).toBe(0);
      expect(state.programs).toHaveLength(1);
      expect(state.programs[0]).toEqual(program);
      expect(state.programFullScoresValSet[0]).toBe(avgScore);
      expect(state.programSubScoresValSet[0]).toEqual(subscores);
    });
    
    it('should handle multiple programs', () => {
      const program1: ComponentMap = { system_prompt: 'Prompt 1' };
      const program2: ComponentMap = { system_prompt: 'Prompt 2' };
      
      state.addProgram(program1, [], 0.5, [], [0.5, 0.5], 0);
      state.addProgram(program2, [0], 0.7, [], [0.7, 0.7], 10);
      
      expect(state.programs).toHaveLength(2);
      expect(state.programParents[1]).toEqual([0]);
    });
  });
  
  describe('isConsistent', () => {
    it('should return true for consistent state', () => {
      const program: ComponentMap = { system_prompt: 'Test' };
      state.addProgram(program, [], 0.5, [], [0.5], 0);
      
      expect(state.isConsistent()).toBe(true);
    });
    
    it('should detect inconsistent state', () => {
      state.programs.push({ system_prompt: 'Test' });
      // Not adding other required arrays
      
      expect(state.isConsistent()).toBe(false);
    });
  });
  
  describe('pareto front computation', () => {
    it('should correctly identify pareto front', () => {
      const prog1: ComponentMap = { system_prompt: 'P1' };
      const prog2: ComponentMap = { system_prompt: 'P2' };
      const prog3: ComponentMap = { system_prompt: 'P3' };
      
      // Program 1: scores [0.5, 0.6]
      state.addProgram(prog1, [], 0.55, [], [0.5, 0.6], 0);
      
      // Program 2: scores [0.7, 0.4] - trades off between examples
      state.addProgram(prog2, [], 0.55, [], [0.7, 0.4], 0);
      
      // Program 3: scores [0.8, 0.9] - dominates both
      state.addProgram(prog3, [], 0.85, [], [0.8, 0.9], 0);
      
      expect(state.programAtParetoFrontValset).toContain(prog3);
    });
  });
});