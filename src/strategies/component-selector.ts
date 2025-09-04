import { ComponentMap } from '../types/index.js';

export interface ComponentSelector {
  select(candidate: ComponentMap, iteration: number): string[];
}

export class RoundRobinReflectionComponentSelector implements ComponentSelector {
  select(candidate: ComponentMap, iteration: number): string[] {
    const componentNames = Object.keys(candidate);
    
    if (componentNames.length === 0) {
      return [];
    }
    
    const selectedIdx = iteration % componentNames.length;
    return [componentNames[selectedIdx]];
  }
}

export class AllComponentsSelector implements ComponentSelector {
  select(candidate: ComponentMap, _iteration: number): string[] {
    return Object.keys(candidate);
  }
}

export class RandomComponentSelector implements ComponentSelector {
  private rng: () => number;
  private numComponents: number;
  
  constructor(numComponents: number = 1, rng?: () => number) {
    this.numComponents = numComponents;
    this.rng = rng || Math.random;
  }
  
  select(candidate: ComponentMap, _iteration: number): string[] {
    const componentNames = Object.keys(candidate);
    
    if (componentNames.length === 0) {
      return [];
    }
    
    const numToSelect = Math.min(this.numComponents, componentNames.length);
    const selected: string[] = [];
    const available = [...componentNames];
    
    for (let i = 0; i < numToSelect; i++) {
      const idx = Math.floor(this.rng() * available.length);
      selected.push(available[idx]);
      available.splice(idx, 1);
    }
    
    return selected;
  }
}