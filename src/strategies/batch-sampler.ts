import { DataInst } from '../types/index.js';

export interface BatchSampler<D = DataInst> {
  sample(trainset: D[], iteration: number): D[];
}

export class EpochShuffledBatchSampler<D = DataInst> implements BatchSampler<D> {
  private minibatchSize: number;
  private rng: () => number;
  private shuffledData?: D[];
  private currentIdx: number = 0;
  private currentEpoch: number = -1;
  
  constructor(minibatchSize: number = 3, rng?: () => number) {
    this.minibatchSize = minibatchSize;
    this.rng = rng || Math.random;
  }
  
  sample(trainset: D[], iteration: number): D[] {
    const epoch = Math.floor(iteration * this.minibatchSize / trainset.length);
    
    if (epoch !== this.currentEpoch || !this.shuffledData) {
      this.shuffledData = this.shuffle([...trainset]);
      this.currentEpoch = epoch;
      this.currentIdx = 0;
    }
    
    const batch: D[] = [];
    for (let i = 0; i < this.minibatchSize; i++) {
      batch.push(this.shuffledData[this.currentIdx]);
      this.currentIdx = (this.currentIdx + 1) % this.shuffledData.length;
    }
    
    return batch;
  }
  
  private shuffle<T>(array: T[]): T[] {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }
}

export class RandomBatchSampler<D = DataInst> implements BatchSampler<D> {
  private batchSize: number;
  private rng: () => number;
  
  constructor(batchSize: number = 3, rng?: () => number) {
    this.batchSize = batchSize;
    this.rng = rng || Math.random;
  }
  
  sample(trainset: D[], _iteration: number): D[] {
    const batch: D[] = [];
    for (let i = 0; i < this.batchSize; i++) {
      const idx = Math.floor(this.rng() * trainset.length);
      batch.push(trainset[idx]);
    }
    return batch;
  }
}

export class SequentialBatchSampler<D = DataInst> implements BatchSampler<D> {
  private batchSize: number;
  private currentIdx: number = 0;
  
  constructor(batchSize: number = 3) {
    this.batchSize = batchSize;
  }
  
  sample(trainset: D[], _iteration: number): D[] {
    const batch: D[] = [];
    for (let i = 0; i < this.batchSize; i++) {
      batch.push(trainset[this.currentIdx]);
      this.currentIdx = (this.currentIdx + 1) % trainset.length;
    }
    return batch;
  }
}