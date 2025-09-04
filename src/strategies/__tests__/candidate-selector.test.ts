import { describe, it, expect, beforeEach } from 'vitest';
import { 
  CurrentBestCandidateSelector, 
  ParetoCandidateSelector,
  RandomCandidateSelector
} from '../candidate-selector.js';
import { GEPAState } from '../../core/state.js';

describe('Candidate Selectors', () => {
  let state: GEPAState;
  
  beforeEach(() => {
    state = new GEPAState();
    
    // Add test programs with different scores
    state.addProgram({ prompt: 'P1' }, [], 0.5, [], [0.5], 0);
    state.addProgram({ prompt: 'P2' }, [0], 0.7, [], [0.7], 0);
    state.addProgram({ prompt: 'P3' }, [1], 0.3, [], [0.3], 0);
  });
  
  describe('CurrentBestCandidateSelector', () => {
    it('should select the program with highest score', () => {
      const selector = new CurrentBestCandidateSelector();
      const [program, indices] = selector.select(state);
      
      expect(program).toEqual({ prompt: 'P2' });
      expect(indices).toEqual([1]);
    });
  });
  
  describe('ParetoCandidateSelector', () => {
    it('should select from pareto front', () => {
      const mockRng = () => 0.5;
      const selector = new ParetoCandidateSelector(mockRng);
      
      // Setup pareto front
      state.programAtParetoFrontValset = [
        { prompt: 'P2' }
      ];
      
      const [program, indices] = selector.select(state);
      
      expect(program).toEqual({ prompt: 'P2' });
      expect(indices[0]).toBeGreaterThanOrEqual(0);
    });
    
    it('should fall back to best when pareto front is empty', () => {
      const selector = new ParetoCandidateSelector();
      state.programAtParetoFrontValset = [];
      
      const [program, indices] = selector.select(state);
      
      expect(program).toEqual({ prompt: 'P2' });
      expect(indices).toEqual([1]);
    });
  });
  
  describe('RandomCandidateSelector', () => {
    it('should select a random program', () => {
      const mockRng = () => 0.3; // Will select index 0
      const selector = new RandomCandidateSelector(mockRng);
      
      const [program, indices] = selector.select(state);
      
      expect(program).toEqual({ prompt: 'P1' });
      expect(indices).toEqual([0]);
    });
  });
});