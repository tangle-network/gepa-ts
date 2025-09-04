import { describe, it, expect } from 'vitest';
import { 
  EpochShuffledBatchSampler,
  RandomBatchSampler,
  SequentialBatchSampler
} from '../batch-sampler.js';

describe('Batch Samplers', () => {
  const trainset = [
    { id: 1, data: 'item1' },
    { id: 2, data: 'item2' },
    { id: 3, data: 'item3' },
    { id: 4, data: 'item4' },
    { id: 5, data: 'item5' }
  ];
  
  describe('EpochShuffledBatchSampler', () => {
    it('should sample minibatch size items', () => {
      const sampler = new EpochShuffledBatchSampler(3);
      const batch = sampler.sample(trainset, 0);
      
      expect(batch).toHaveLength(3);
    });
    
    it('should reshuffle at new epoch', () => {
      const sampler = new EpochShuffledBatchSampler(2);
      
      // First epoch samples
      const batch1 = sampler.sample(trainset, 0);
      const batch2 = sampler.sample(trainset, 1);
      const batch3 = sampler.sample(trainset, 2);
      
      // Should have seen 6 items (may have repeats due to epoch boundary)
      expect(batch1).toHaveLength(2);
      expect(batch2).toHaveLength(2);
      expect(batch3).toHaveLength(2);
    });
    
    it('should use provided RNG for shuffling', () => {
      let seed = 42;
      const mockRng = () => {
        const x = Math.sin(seed++) * 10000;
        return x - Math.floor(x);
      };
      
      const sampler1 = new EpochShuffledBatchSampler(2, mockRng);
      seed = 42; // Reset seed
      const sampler2 = new EpochShuffledBatchSampler(2, mockRng);
      
      const batch1 = sampler1.sample(trainset, 0);
      seed = 42; // Reset seed again
      const batch2 = sampler2.sample(trainset, 0);
      
      expect(batch1).toEqual(batch2);
    });
  });
  
  describe('RandomBatchSampler', () => {
    it('should sample random items', () => {
      const sampler = new RandomBatchSampler(3);
      const batch = sampler.sample(trainset, 0);
      
      expect(batch).toHaveLength(3);
      batch.forEach(item => {
        expect(trainset).toContainEqual(item);
      });
    });
    
    it('should allow duplicates', () => {
      const mockRng = () => 0.1; // Always returns same index
      const sampler = new RandomBatchSampler(3, mockRng);
      const batch = sampler.sample(trainset, 0);
      
      expect(batch).toHaveLength(3);
      expect(batch[0]).toEqual(batch[1]);
      expect(batch[1]).toEqual(batch[2]);
    });
  });
  
  describe('SequentialBatchSampler', () => {
    it('should sample sequentially', () => {
      const sampler = new SequentialBatchSampler(2);
      
      const batch1 = sampler.sample(trainset, 0);
      const batch2 = sampler.sample(trainset, 1);
      const batch3 = sampler.sample(trainset, 2);
      
      expect(batch1).toEqual([trainset[0], trainset[1]]);
      expect(batch2).toEqual([trainset[2], trainset[3]]);
      expect(batch3).toEqual([trainset[4], trainset[0]]); // Wraps around
    });
    
    it('should wrap around when reaching end', () => {
      const sampler = new SequentialBatchSampler(3);
      
      const batch1 = sampler.sample(trainset, 0);
      const batch2 = sampler.sample(trainset, 1);
      
      expect(batch1).toEqual([trainset[0], trainset[1], trainset[2]]);
      expect(batch2).toEqual([trainset[3], trainset[4], trainset[0]]);
    });
  });
});