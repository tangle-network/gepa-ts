/**
 * Tests for mutation and merge proposers
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import { 
  MutationProposer,
  MergeProposer,
  DSPyInstructionProposer 
} from '../../src/proposer/index.js';
import { TestLLM } from '../infrastructure/record-replay.js';

describe('Proposers', () => {
  let testLLM: TestLLM;
  
  beforeEach(() => {
    testLLM = new TestLLM({ mode: 'replay' });
  });
  
  describe('MutationProposer', () => {
    it('should mutate text components', async () => {
      const proposer = new MutationProposer(
        async (prompt: string) => testLLM.generate(prompt)
      );
      
      const candidate = {
        instruction: 'Original instruction text',
        format: 'Original format text'
      };
      
      const reflectiveDataset = {
        instruction: [
          {
            Score: 0.3,
            Feedback: 'Too vague',
            'Current Text': 'Original instruction text'
          }
        ]
      };
      
      const mutated = await proposer.propose(
        candidate,
        reflectiveDataset,
        ['instruction']
      );
      
      expect(mutated).toHaveProperty('instruction');
      expect(mutated.instruction).not.toBe(candidate.instruction);
      expect(mutated.format).toBe(candidate.format); // Unchanged
    });
    
    it('should handle multiple components', async () => {
      const proposer = new MutationProposer(
        async (prompt: string) => `Improved: ${prompt.slice(0, 20)}`
      );
      
      const candidate = {
        comp1: 'Text 1',
        comp2: 'Text 2',
        comp3: 'Text 3'
      };
      
      const reflectiveDataset = {
        comp1: [{ Score: 0.4, Feedback: 'Improve' }],
        comp2: [{ Score: 0.5, Feedback: 'Enhance' }]
      };
      
      const mutated = await proposer.propose(
        candidate,
        reflectiveDataset,
        ['comp1', 'comp2']
      );
      
      expect(mutated.comp1).toContain('Improved');
      expect(mutated.comp2).toContain('Improved');
      expect(mutated.comp3).toBe('Text 3'); // Unchanged
    });
    
    it('should use feedback to guide mutations', async () => {
      let capturedPrompt = '';
      
      const proposer = new MutationProposer(
        async (prompt: string) => {
          capturedPrompt = prompt;
          return 'Mutated text addressing feedback';
        }
      );
      
      const candidate = {
        text: 'Original text'
      };
      
      const reflectiveDataset = {
        text: [
          {
            Score: 0.2,
            Feedback: 'Needs more clarity and specific examples',
            Problem: 'Test problem',
            'Expected Output': 'Expected result'
          }
        ]
      };
      
      await proposer.propose(candidate, reflectiveDataset, ['text']);
      
      // Check that feedback was included in mutation prompt
      expect(capturedPrompt).toContain('clarity');
      expect(capturedPrompt).toContain('examples');
      expect(capturedPrompt).toContain('Original text');
    });
  });
  
  describe('MergeProposer', () => {
    it('should merge two candidates', async () => {
      const proposer = new MergeProposer(
        async (prompt: string) => 'Merged text combining best of both'
      );
      
      const candidate1 = {
        instruction: 'Be precise',
        format: 'JSON output'
      };
      
      const candidate2 = {
        instruction: 'Be clear',
        format: 'XML output'
      };
      
      const merged = await proposer.propose(
        candidate1,
        candidate2,
        ['instruction', 'format']
      );
      
      expect(merged).toHaveProperty('instruction');
      expect(merged).toHaveProperty('format');
      expect(merged.instruction).toContain('Merged');
      expect(merged.format).toContain('Merged');
    });
    
    it('should only merge specified components', async () => {
      const proposer = new MergeProposer(
        async () => 'Merged content'
      );
      
      const candidate1 = {
        comp1: 'A1',
        comp2: 'A2',
        comp3: 'A3'
      };
      
      const candidate2 = {
        comp1: 'B1',
        comp2: 'B2',
        comp3: 'B3'
      };
      
      const merged = await proposer.propose(
        candidate1,
        candidate2,
        ['comp1']
      );
      
      expect(merged.comp1).toBe('Merged content');
      expect(merged.comp2).toBe('A2'); // From candidate1
      expect(merged.comp3).toBe('A3'); // From candidate1
    });
    
    it('should handle ancestor-aware merging', async () => {
      let capturedPrompt = '';
      
      const proposer = new MergeProposer(
        async (prompt: string) => {
          capturedPrompt = prompt;
          return 'Ancestor-aware merge';
        }
      );
      
      const candidate1 = {
        text: 'Version A with feature X'
      };
      
      const candidate2 = {
        text: 'Version B with feature Y'
      };
      
      const commonAncestor = {
        text: 'Original version'
      };
      
      const merged = await proposer.propose(
        candidate1,
        candidate2,
        ['text'],
        commonAncestor
      );
      
      expect(merged.text).toBe('Ancestor-aware merge');
      expect(capturedPrompt).toContain('Original version');
      expect(capturedPrompt).toContain('Version A');
      expect(capturedPrompt).toContain('Version B');
    });
    
    it('should implement doesTripletHaveDesirablePredictors', () => {
      const proposer = new MergeProposer(async () => '');
      
      const candidate1 = {
        comp1: 'Text A',
        comp2: 'Common'
      };
      
      const candidate2 = {
        comp1: 'Text B',
        comp2: 'Common'
      };
      
      const ancestor = {
        comp1: 'Original',
        comp2: 'Common'
      };
      
      // comp1 differs between candidates but not comp2
      const hasDesirable = proposer.doesTripletHaveDesirablePredictors(
        candidate1,
        candidate2,
        ancestor,
        ['comp1', 'comp2']
      );
      
      expect(hasDesirable).toBe(true);
      
      // All same - no desirable predictors
      const noDesirable = proposer.doesTripletHaveDesirablePredictors(
        candidate1,
        { comp1: 'Text A', comp2: 'Common' },
        candidate1,
        ['comp1', 'comp2']
      );
      
      expect(noDesirable).toBe(false);
    });
  });
  
  describe('DSPyInstructionProposer', () => {
    it('should propose improved instructions based on traces', async () => {
      const proposer = new DSPyInstructionProposer(
        async (prompt: string) => {
          if (prompt.includes('failure')) {
            return 'Be more specific and provide clear output format. Show your reasoning step by step.';
          }
          return 'Improved instruction';
        }
      );
      
      const currentInstruction = 'Answer the question';
      
      const datasetWithFeedback = [
        {
          Score: 0.2,
          Feedback: 'Output format was incorrect',
          'Generated Outputs': { answer: 'wrong format' }
        },
        {
          Score: 0.9,
          Feedback: 'Good answer with clear reasoning',
          'Generated Outputs': { answer: 'correct', reasoning: 'step by step' }
        },
        {
          Score: 0.3,
          Feedback: 'Missing required fields',
          'Generated Outputs': { partial: 'incomplete' }
        }
      ];
      
      const improved = await proposer.proposeInstruction(
        currentInstruction,
        datasetWithFeedback
      );
      
      expect(improved).not.toBe(currentInstruction);
      expect(improved).toContain('specific');
      expect(improved).toContain('format');
      expect(improved.length).toBeGreaterThan(20);
    });
    
    it('should analyze traces to identify patterns', async () => {
      const proposer = new DSPyInstructionProposer(
        async () => 'Improved based on analysis'
      );
      
      const datasetWithFeedback = [
        { Score: 0.1, Feedback: 'Wrong format' },
        { Score: 0.2, Feedback: 'Missing output fields' },
        { Score: 0.15, Feedback: 'Incorrect reasoning' },
        { Score: 0.9, Feedback: 'Perfect', 'Generated Outputs': { good: 'example' } },
        { Score: 0.85, Feedback: 'Very good' }
      ];
      
      const analysis = (proposer as any).analyzeTraces(datasetWithFeedback);
      
      expect(analysis.failurePatterns).toContain('Wrong format');
      expect(analysis.failurePatterns).toContain('Missing output fields');
      expect(analysis.commonErrors).toContain('format_error');
      expect(analysis.commonErrors).toContain('missing_output');
      expect(analysis.recommendations).toContain('Specify output format explicitly');
      expect(analysis.successPatterns).toHaveLength(2);
    });
    
    it('should generate multiple instruction variants', async () => {
      let callCount = 0;
      
      const proposer = new DSPyInstructionProposer(
        async () => {
          callCount++;
          return `Variant ${callCount} instruction`;
        }
      );
      
      const currentInstruction = 'Base instruction';
      const datasetWithFeedback = [
        { Score: 0.5, Feedback: 'Average performance' }
      ];
      
      const variants = await proposer.proposeVariants(
        currentInstruction,
        datasetWithFeedback,
        3
      );
      
      expect(variants).toHaveLength(3);
      expect(new Set(variants).size).toBeGreaterThan(1); // Should be different
      
      // Check that variants have structure
      for (const variant of variants) {
        expect(variant.length).toBeGreaterThan(0);
      }
    });
    
    it('should enhance instruction locally when LLM fails', async () => {
      const proposer = new DSPyInstructionProposer(
        async () => '' // Return empty string to trigger fallback
      );
      
      const currentInstruction = 'Simple instruction';
      const datasetWithFeedback = [
        { Score: 0.2, Feedback: 'Format error and missing output' }
      ];
      
      const improved = await proposer.proposeInstruction(
        currentInstruction,
        datasetWithFeedback
      );
      
      expect(improved).not.toBe(''); // Should use fallback
      expect(improved).toContain('Simple instruction'); // Should build on current
      expect(improved).toContain('format'); // Should address errors
      expect(improved.length).toBeGreaterThan(currentInstruction.length);
    });
  });
  
  describe('Integration', () => {
    it('should combine mutation and merge effectively', async () => {
      const mutator = new MutationProposer(
        async (prompt: string) => `Mutated: ${prompt.slice(0, 10)}`
      );
      
      const merger = new MergeProposer(
        async (prompt: string) => {
          // Extract both texts and combine
          if (prompt.includes('Mutated:')) {
            return 'Combined mutated texts';
          }
          return 'Merged';
        }
      );
      
      const original = {
        text: 'Original text content'
      };
      
      // First mutation
      const mutated1 = await mutator.propose(
        original,
        { text: [{ Score: 0.5, Feedback: 'Improve' }] },
        ['text']
      );
      
      // Second mutation
      const mutated2 = await mutator.propose(
        original,
        { text: [{ Score: 0.4, Feedback: 'Enhance' }] },
        ['text']
      );
      
      // Merge mutations
      const merged = await merger.propose(
        mutated1,
        mutated2,
        ['text']
      );
      
      expect(mutated1.text).toContain('Mutated');
      expect(mutated2.text).toContain('Mutated');
      expect(merged.text).toBe('Combined mutated texts');
    });
  });
});