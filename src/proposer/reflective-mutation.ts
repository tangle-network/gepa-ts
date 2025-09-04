import { 
  ComponentMap, 
  DataInst, 
  LanguageModel, 
  LoggerProtocol,
  ProposalResult,
  ReflectiveDataset,
  RolloutOutput,
  Trajectory
} from '../types/index.js';
import { GEPAAdapter } from '../core/adapter.js';
import { GEPAState } from '../core/state.js';
import { CandidateSelector } from '../strategies/candidate-selector.js';
import { ComponentSelector } from '../strategies/component-selector.js';
import { BatchSampler } from '../strategies/batch-sampler.js';
import { proposeInstructionWithLM } from './instruction-proposal.js';

export interface ReflectiveMutationProposerConfig<D = DataInst> {
  logger: LoggerProtocol;
  trainset: D[];
  adapter: GEPAAdapter<D>;
  candidateSelector: CandidateSelector;
  componentSelector: ComponentSelector;
  batchSampler: BatchSampler<D>;
  perfectScore: number;
  skipPerfectScore: boolean;
  useWandB?: boolean;
  reflectionLM?: LanguageModel;
}

export class ReflectiveMutationProposer<D = DataInst, T = Trajectory, O = RolloutOutput> {
  private config: ReflectiveMutationProposerConfig<D>;
  
  constructor(config: ReflectiveMutationProposerConfig<D>) {
    this.config = config;
  }
  
  private async proposeNewTexts(
    candidate: ComponentMap,
    reflectiveDataset: ReflectiveDataset,
    componentsToUpdate: string[]
  ): Promise<ComponentMap> {
    if (this.config.adapter.proposeNewTexts) {
      const result = this.config.adapter.proposeNewTexts(
        candidate,
        reflectiveDataset,
        componentsToUpdate
      );
      return result instanceof Promise ? await result : result;
    }
    
    if (!this.config.reflectionLM) {
      throw new Error('Reflection LM required when adapter does not provide proposeNewTexts');
    }
    
    const newTexts: ComponentMap = {};
    
    for (const name of componentsToUpdate) {
      const baseInstruction = candidate[name];
      const datasetWithFeedback = reflectiveDataset[name];
      
      newTexts[name] = await proposeInstructionWithLM(
        this.config.reflectionLM,
        baseInstruction,
        datasetWithFeedback
      );
    }
    
    return newTexts;
  }
  
  async propose(state: GEPAState): Promise<ProposalResult | null> {
    const i = state.i + 1;
    
    const [currentProgram, parentIds] = this.config.candidateSelector.select(state);
    const currentProgramId = parentIds[0];
    
    state.fullProgramTrace[state.fullProgramTrace.length - 1].selectedProgramCandidate = currentProgramId;
    
    this.config.logger.log(
      `Iteration ${i}: Selected program ${currentProgramId} score: ${state.programFullScoresValSet[currentProgramId]}`
    );
    
    const minibatch = this.config.batchSampler.sample(this.config.trainset, i - 1);
    
    const evalCurrent = await this.config.adapter.evaluate(minibatch, currentProgram, true);
    
    if (!evalCurrent.trajectories || evalCurrent.trajectories.length === 0) {
      this.config.logger.log(`Iteration ${i}: No trajectories captured. Skipping.`);
      return null;
    }
    
    state.totalNumEvals += minibatch.length;
    state.fullProgramTrace[state.fullProgramTrace.length - 1].subsampleScores = evalCurrent.scores;
    
    if (this.config.skipPerfectScore && 
        evalCurrent.scores.every(s => s >= this.config.perfectScore)) {
      this.config.logger.log(`Iteration ${i}: All subsample scores perfect. Skipping.`);
      return null;
    }
    
    const componentsToUpdate = this.config.componentSelector.select(currentProgram, i);
    
    try {
      const reflectiveDataset = await this.config.adapter.makeReflectiveDataset(
        currentProgram,
        evalCurrent,
        componentsToUpdate
      );
      
      const newTexts = await this.proposeNewTexts(
        currentProgram,
        reflectiveDataset,
        componentsToUpdate
      );
      
      for (const [name, text] of Object.entries(newTexts)) {
        this.config.logger.log(`Iteration ${i}: Proposed new text for ${name}: ${text}`);
      }
      
      const newCandidate = { ...currentProgram };
      for (const [name, text] of Object.entries(newTexts)) {
        if (!(name in newCandidate)) {
          throw new Error(`${name} missing in candidate`);
        }
        newCandidate[name] = text;
      }
      
      const evalNew = await this.config.adapter.evaluate(minibatch, newCandidate, false);
      state.totalNumEvals += minibatch.length;
      state.fullProgramTrace[state.fullProgramTrace.length - 1].newSubsampleScores = evalNew.scores;
      
      return {
        candidate: newCandidate,
        tag: 'reflective',
        parentProgramIds: parentIds,
        subsampleScoresBefore: evalCurrent.scores,
        subsampleScoresAfter: evalNew.scores
      };
      
    } catch (error) {
      this.config.logger.log(`Iteration ${i}: Exception during reflection/proposal: ${error}`);
      if (error instanceof Error) {
        this.config.logger.log(error.stack || '');
      }
      return null;
    }
  }
}