**Neuromorphic Graph Architecture**

Implementation Roadmap v4

You, AI, and Eventually a Mathematician

De-risked plan with explicit work allocation across AI, collaborative, and solo tasks

*Work Allocation Legend:*

BLUE = AI Solo --- Claude / Claude Code writes it, you review

ORANGE = Collaborative --- You direct, AI assists, you decide

RED = You Solo --- Run experiments, watch dynamics, build intuition, have insights

*Risk Legend:*

RED TEXT = Unknown

GREEN = Validation

DARK RED = Kill criterion

PURPLE TEXT = Pivot signal

What We're Building

A self-organising directed graph that computes the way a brain computes. Not a simulation of the brain. Not a spiking neural network. Not a deep learning model with biological window dressing. A graph --- nodes and edges --- whose topology, connection types, and dynamics are derived from the computational principles the brain actually uses.

The core insight: the brain is literally a directed graph with cycles. Neurons are nodes. Synapses are directed weighted edges. The graph self-organises its own topology through plasticity. Discrete attractor states crystallise from continuous dynamics. Everything else is detail on top of this.

The Graph

A connected graph with spatial metric. Nodes have positions. Connection probability, delay, and strength are functions of distance. Edges are directed, weighted, and typed. The graph supports dynamic topology --- edges are created and destroyed at runtime based on activity.

Three Edge Types

**Chemical directed edges.** Weighted, with stochastic release probability, short-term facilitation/depression dynamics, and distance-dependent conduction delays. Subtyped as: driving (feedforward, targets basal compartment), modulatory (feedback, targets apical compartment), inhibitory-perisomatic (PV-like, scales output gain), inhibitory-dendritic (SST-like, gates apical input).

**Electrical bidirectional edges.** Gap junctions between PV-type inhibitory nodes. Near-instantaneous, non-plastic, symmetric. Enable gamma-frequency synchrony.

**Retrograde edges.** Post-to-presynaptic suppression. Activity-dependent, target node suppresses incoming edge transmission. Edge-level gain control.

Two-Compartment Nodes

Each excitatory node has a basal compartment (integrates driving input --- bottom-up evidence) and an apical compartment (integrates modulatory input --- top-down predictions/context). The apical compartment gates the basal compartment's output:

*output = f(basal) × g(apical) + noise*

This is the minimal dendritic computation: one layer of nonlinear gating that captures the prediction-evidence interaction. When no top-down signal arrives, the node still functions (ungated). When top-down predictions are strong, they amplify or suppress the node's response. Each node also has learnable intrinsic parameters (threshold, gain) that adapt homeostatically.

Four Node Types

**Excitatory (\~80%):** Two-compartment, project long-range, carry representational content. Split into representation nodes (generate predictions, send modulatory edges downward) and error nodes (compute prediction -- evidence discrepancy, send driving edges upward).

**PV interneurons (\~7%):** Fast perisomatic inhibition. Gap-junction-coupled. Drive gamma oscillations via PING mechanism. Implement divisive normalisation (gain control) and enforce sparsity through winner-take-all competition.

**SST interneurons (\~7%):** Dendritic inhibition targeting apical compartments. Gate top-down input. Mediate expectation suppression --- when active, they suppress the influence of predictions, silencing responses to expected/predictable input.

**VIP interneurons (\~6%):** Inhibit SST cells, creating disinhibition. When activated by long-range signals or arousal, they release excitatory nodes from SST gating, opening them to feedback. This is the surprise/attention circuit.

Core motif: VIP → suppresses SST → suppresses Excitatory apical. PV → suppresses Excitatory soma.

The Core Algorithm: Predictive Coding

The graph implements hierarchical Bayesian inference. Higher levels generate predictions about what lower levels should be doing. Lower levels compute the discrepancy (prediction error). Errors flow upward and update the higher-level representation, which generates better predictions, which suppress the errors. This converges on a representation that best explains the input given the graph's internal model.

Precision weighting gates which errors matter: high-precision errors (reliable signals) drive strong updating; low-precision errors are suppressed. This is how attention works --- increasing the precision of task-relevant error signals.

Five Learning Subgraphs

The full graph contains five architecturally distinct subgraphs, each running a different learning algorithm:

  --------------- ------------------------ ------------------------------- -----------------------------------------------------
  **Subgraph**    **Algorithm**            **What It Learns**              **Key Architecture**

  Cortex          Predictive coding        Generative model of input       Hierarchical, bidirectional, typed edges

  Basal ganglia   Reinforcement learning   Action values and selection     Direct/indirect/hyperdirect pathway competition

  Cerebellum      Supervised learning      Forward models and timing       Feedforward, regular, climbing fibre error

  Hippocampus     One-shot Hebbian/BTSP    Episodic memories               Sparse expansion → recurrent attractor → comparator

  Amygdala        Pavlovian conditioning   Stimulus-valence associations   Fast subcortical shortcut from thalamus
  --------------- ------------------------ ------------------------------- -----------------------------------------------------

All five interact through a thalamic routing hub that dynamically gates which subgraphs communicate at any moment.

Six Plasticity Timescales

**Release probability (ms):** Short-term facilitation/depression at individual edges. Edges have internal state that makes them dynamic filters.

**BTSP (seconds):** Single-trial dendritic-plateau-driven plasticity. One-shot episodic encoding in the hippocampal subgraph.

**STDP (minutes--hours):** Spike-timing-dependent plasticity. Pre-before-post strengthens, post-before-pre weakens. Learns causal/predictive relationships.

**Intrinsic plasticity (hours):** Node-level excitability changes independent of edge weights. Homeostatic.

**Structural plasticity (days--weeks):** Edge creation and pruning. The graph topology itself reorganises.

**Metaplasticity (overlaid):** BCM sliding threshold --- the plasticity rules themselves adapt based on recent activity, preventing saturation.

On recall, stored patterns undergo temporary destabilisation and reconsolidation. Retrieval is reconstructive, not reproductive.

Temporal Coordination

The graph operates in time through oscillatory dynamics. Gamma oscillations (30--150 Hz) driven by PV interneurons bind features and gate local computation. Theta oscillations (4--8 Hz) coordinate cross-area communication and sequence working memory items. Cross-frequency coupling (gamma nested in theta) provides temporal multiplexing. Alpha oscillations gate the thalamic routing hub. The system alternates between online mode (processing external input) and offline mode (replay and consolidation).

Neuromodulation

Four neuromodulatory subgraphs (not just scalar signals) with their own recurrent dynamics broadcast control signals across the graph. Dopamine: reward prediction error, precision, learning rate. Norepinephrine: arousal, global gain, exploration-exploitation. Acetylcholine: feedforward-feedback balance, plasticity gating. Serotonin: temporal discounting. Astrocytes provide additional local-to-regional slow modulation.

The Constraint That Ties It Together

Energy budget as an explicit optimisation constraint. Total activity, edge count, and weight magnitude are penalised. Many properties in this stack --- sparsity, predictive coding, pruning, efficient coding --- are expected to emerge as consequences of optimising under this metabolic constraint rather than needing to be independently engineered. This is the deepest hypothesis of the project: that the right energy functional on the right substrate produces brain-like computation for free.

The Complete Stack

17 layers. 5 learning algorithms. 3 edge types. 2 node compartments. 6 plasticity timescales. 4 neuromodulatory systems. 4 attractor types (point, line, limit cycle, transient). E/I balance tuned to near-criticality. Hierarchical organisation with online/offline modes. Efference copy for self-generated action prediction.

The roadmap that follows describes how to build this incrementally, testing existential risks as early as possible, with explicit criteria for when to continue, pivot, or stop.

Design Philosophy: Front-Load the Existential Risks

This roadmap is ordered by risk, not by logical dependency. At every phase, we identify the question that would be most expensive to get wrong and answer it before investing further. The core principle:

**Never spend more than 3 months building something whose foundational assumption hasn't been tested.**

This creates a branching plan with explicit kill criteria and pivot signals. At the end of each phase, you make a go/no-go/pivot decision before proceeding. If the project dies, it dies cheap.

The Three Existential Questions

These are the questions that, if answered negatively, mean the entire project needs fundamental rethinking. They must be answered as early as possible:

**Q1: Can predictive coding self-organise on a graph substrate?** If not, the entire thesis --- that brain-like computation emerges from self-organising graph dynamics --- is wrong. Tested in Phase 1.

**Q2: Does the energy constraint produce useful self-organisation, or do you need hand-building?** This determines whether you're building 17 layers by hand or discovering them from first principles. Tested in Phase 1, in parallel with Q1.

**Q3: Can multiple learning systems (predictive coding + RL + one-shot) coexist on the same graph without destabilising each other?** If not, the multi-subgraph architecture doesn't work. Tested in Phase 3, before you invest in building each subgraph at full fidelity.

Phase 0: The Substrate (Month 1--2)

*Pure infrastructure. No science. Maximum AI leverage.*

0.1 Graph Data Structure

> **🤖 AI SOLO:** Write the core graph class: nodes with positions, typed directed edges with weight/delay/state, adjacency structure optimised for message passing. Claude Code can generate this from a spec. You review the API and data layout.
>
> **🤖 AI SOLO:** Distance-dependent connection probability function. Standard exponential decay. I give you the functional form and typical parameter ranges from the literature (Hellwig 2000, Perin et al. 2011). Claude Code implements it.
>
> **🤖 AI SOLO:** STDP implementation. The classic asymmetric exponential window is fully specified mathematically. I give you the equations (Bi & Poo 1998, Song et al. 2000), Claude Code implements with configurable time constants and learning rate.
>
> **🤖 AI SOLO:** Homeostatic weight normalisation. Synaptic scaling is a simple renormalisation step. Equations are standard (Turrigiano 2008). Claude Code writes it.
>
> **🤖 AI SOLO:** Visualisation. Real-time graph rendering showing node activations, edge weights, cell types. This is pure frontend engineering. Claude Code.

0.2 Node Model

> **🤝 COLLAB:** Design the two-compartment node update equations. I propose the math, you interrogate whether it captures what you need. We iterate on paper before coding. Key decisions: activation function shape, gating function form, noise model. I give you options with tradeoffs, you choose.
>
> **🤖 AI SOLO:** Implement the chosen node model. Once we've agreed the equations, Claude Code writes the vectorised JAX implementation.
>
> **🤝 COLLAB:** Cell type connectivity motif (PV/SST/VIP wiring). I provide the biological connectivity rules from Tremblay et al. 2016 and Wang & Yang 2018. You verify they make sense for your architecture. We agree on the implementation. Claude Code writes it.

0.3 Validation

> **🤖 AI SOLO:** Parameter sweep infrastructure. Grid search or random search over E/I ratio, STDP learning rate, noise level. Logging, checkpointing, analysis plots. Pure engineering. Claude Code.
>
> **🧠 YOU SOLO:** Run the sweeps. Watch the dynamics at different parameter settings. Develop your first intuitions about how this system behaves. What does runaway excitation look like on your graph? What does pathological silence look like? What does the edge of stability feel like? This is where you start becoming the world expert on your system. Nobody can do this for you.
>
> **🤝 COLLAB:** Diagnose instabilities. If the network explodes or goes silent, describe the symptoms to me. I'll reason about the likely cause based on known E/I balance theory and suggest parameter adjustments. You test them.
>
> **KILL CRITERION:** If no stable parameter regime exists at N=5000 after 2 months, simplify to single-compartment nodes. Get those stable. Re-add apical compartment incrementally.

**Phase 0 work split:** \~60% AI solo, \~25% collaborative, \~15% you solo. This is the most AI-leveraged phase.

Phase 1A: Hand-Built Predictive Coding (Month 3--6)

*The first existential test. The science starts here.*

1A.1 Architecture Setup

> **🤝 COLLAB:** Design the hierarchical wiring between two levels. I provide the Bastos et al. 2012 canonical microcircuit for predictive coding: error nodes in L2/3 project feedforward (driving), representation nodes in L5 project feedback (modulatory). We map this onto your graph's node types and edge types. You decide the specific node-to-level assignment strategy.
>
> **🤖 AI SOLO:** Implement the hierarchical wiring on the graph. Given the agreed design, Claude Code generates the inter-level connectivity with correct edge types, targeting the right compartments.
>
> **🤖 AI SOLO:** Implement the predictive coding update rule. The error computation (ε = basal - apical) and representation update (Δμ = α·π·ε) are mathematically simple. I give you the equations from Bogacz 2017. Claude Code implements them.

1A.2 The Critical Unknown: STDP-PC Interaction

> **🤝 COLLAB:** Design the experiment to test whether STDP helps or hurts PC. I propose: run PC with STDP on, measure prediction error over time. Run PC with STDP off (fixed weights), measure prediction error. Run PC with STDP on only modulatory edges. Run PC with three-factor STDP (gated by error sign). Compare all four conditions. You decide the input sequences, network size, and duration.
>
> **🤖 AI SOLO:** Implement the four experimental conditions. Code the three-factor STDP variant (I provide the gating equations from Fremaux & Gerstner 2016). Set up the comparison pipeline with statistical tests. Claude Code.
>
> **🧠 YOU SOLO:** Run the experiments. This will take days of compute time during which you watch, think, and develop intuitions. When the results come in, you're the one who looks at the learning curves and decides: is STDP helping? Is three-factor STDP better? Is the improvement meaningful or marginal? These are judgment calls that determine the entire future direction.
>
> **UNKNOWN:** The STDP-PC interaction. Nobody knows if they cooperate on a self-organising graph. This is the question that determines whether the hand-built track succeeds. Your experiment answers it.

1A.3 Analysis and Interpretation

> **🤖 AI SOLO:** Compute the standard diagnostics: prediction error time series, sparsity statistics, information flow (transfer entropy between levels), weight distribution histograms, cell-type-specific firing rate statistics. All standard analysis code. Claude Code.
>
> **🤝 COLLAB:** Interpret the mismatch negativity results. I can tell you what the expected signature looks like (increased error node activity at the violated position, decreased representation node confidence). You tell me what you actually see. Together we figure out whether deviations from the expected pattern are bugs, parameter issues, or interesting new phenomena.
>
> **🧠 YOU SOLO:** The judgment call: does this constitute "predictive coding on a self-organising graph"? Is the mismatch response strong enough to be meaningful? Is the prediction quality good enough to be useful? Does this actually work? You decide. This is your project.
>
> **VALIDATION:** Mismatch negativity: error node activity spikes on prediction violation. Prediction quality exceeds trivial baseline. Both must pass.
>
> **KILL CRITERION:** If no STDP variant produces stable predictive coding after 6 weeks of parameter tuning, see Failure Mode 3 in Appendix B.

**Phase 1A work split:** \~35% AI solo, \~30% collaborative, \~35% you solo. The balance shifts toward you because the science starts.

Phase 1B: Energy Self-Organisation Experiment (Month 3--6, parallel)

*Runs simultaneously with 1A. The most theoretically interesting experiment in the project.*

1B.1 Setup

> **🤝 COLLAB:** Design the energy functional. I provide the standard options: L1 on activity (sparsity), L2 on weights (regularisation), L1 on edge count (pruning). We discuss what performance objective to pair with the energy penalty: reconstruction error vs prediction error vs mutual information. I lay out the theoretical implications of each choice. You decide.
>
> **🤖 AI SOLO:** Implement the energy-constrained training loop. The energy terms are simple math. The tricky bit is allowing edge types to change (driving ↔ modulatory) via structural plasticity. I design the type-switching rule; Claude Code implements it.
>
> **🤖 AI SOLO:** Set up the λ sweep. The relative weights λ1, λ2, λ3 of the energy terms are hyperparameters. Set up a grid search. Claude Code.

1B.2 The Deep Questions

> **🧠 YOU SOLO:** Run the energy-constrained network. Watch what happens. This is the most intellectually exciting part of the entire project. Does the network self-organise into something recognisable? Do you see hierarchy emerging? Do error-like and representation-like node roles differentiate? Do edge types self-sort into driving-up and modulatory-down? Nobody knows what will happen. Your eyes are the first to see it. Take notes. Screenshot everything. Record your observations obsessively.
>
> **🤝 COLLAB:** If interesting structure emerges, the question becomes: WHY? Work with AI to perform Lyapunov stability analysis of the self-organised equilibria, information-theoretic characterisation of what the energy functional is actually optimising, and connection to variational inference (is the energy functional an approximation to variational free energy?). I can walk you through each of these analytical frameworks step by step and help you apply them to your specific system. The formalisations are standard --- the novelty is applying them to YOUR graph.
>
> **UNKNOWN:** Whether any form of predictive coding emerges from energy minimisation alone on a self-organising graph. This is genuinely unknown. Zhang et al. 2025 showed it in a fixed-topology spiking network. Your substrate is different. Original science.
>
> **🤝 COLLAB:** Compare the emergent structure to the hand-built PC network from Track 1A. I help you design the comparison metrics: representation similarity analysis, information flow profiles, sparsity patterns, weight structure. Are they converging on the same solution from different starting points? Or are they qualitatively different? This comparison is the core result of Phase 1.
>
> **PIVOT SIGNAL:** If energy self-organisation produces predictive coding, this becomes the main story. Pivot hard.

**Phase 1B work split:** \~25% AI solo, \~30% collaborative, \~45% you solo. This is where the deep thinking lives.

Phase 1 Decision Gate

At the end of Phase 1 (month 6), you are at the first major decision point. Based on the outcomes of tracks 1A and 1B, one of four paths:

  --------------------- ----------------------- ------------------------------------------------------------------ -------------------------------------------------------------
  **1A (Hand-Built)**   **1B (Energy)**         **Interpretation**                                                 **Action**

  PC works              PC emerges              Energy constraint is sufficient. Self-org is the story.            Pivot to energy-first. Best possible outcome.

  PC works              Partial self-org only   Energy helps but doesn't suffice. Hybrid approach.                 Proceed with hand-built + energy constraint as regulariser.

  PC works              Nothing useful          Energy constraint wrong or insufficient. Hand-build is the path.   Proceed hand-built. Revisit energy later.

  PC fails              Any                     Graph substrate can't do PC with current node model.               STOP. Fix node model. Do not proceed.
  --------------------- ----------------------- ------------------------------------------------------------------ -------------------------------------------------------------

**Maximum sunk cost if the project dies at Phase 1: 6 months.**

Phase 2: Oscillatory Dynamics (Month 7--10)

2.1 Implementation

> **🤖 AI SOLO:** Implement PV gap junction coupling. Bidirectional, symmetric, non-plastic electrical edges between PV nodes within a spatial radius. The coupling equation is simple (resistive current proportional to voltage difference). I give you the biophysics; Claude Code implements it.
>
> **🤝 COLLAB:** Design the theta drive. Options: (a) external sinusoidal modulation of excitability, (b) emergent from SST network resonance, (c) a dedicated pacemaker node population. I explain the computational implications of each. You choose based on what you learned in Phase 1 about your network's dynamics.
>
> **🤖 AI SOLO:** Power spectral analysis pipeline for detecting gamma, theta, and cross-frequency coupling. Modulation index computation. Phase-amplitude coupling statistics. Standard signal processing. Claude Code.

2.2 The Oscillation-PC Interaction

> **🧠 YOU SOLO:** Turn on oscillatory dynamics in the network that's doing predictive coding. Watch what happens. Does gamma synchrony sharpen or blur the prediction error signals? Does theta create natural windows for feedforward vs feedback processing? Or does oscillatory modulation of excitability disrupt the inference loop? This is an observational phase. You need to develop intuition for how oscillations and inference interact in YOUR system, not in theory.
>
> **🤝 COLLAB:** Debug frequency problems. If gamma is at the wrong frequency, I can reason analytically about the relationship between PV-pyramidal coupling strength, GABA decay time, excitatory drive, and gamma frequency (Börgers & Kopell 2003). I suggest parameter adjustments; you test them.

2.3 Temporal Binding

> **🧠 YOU SOLO:** Present multiple distinct input patterns simultaneously. Observe whether they segregate into different gamma phases. This is the binding test. If it works, you have temporal multiplexing on a self-organising graph. If it doesn't, you need to understand why --- is it a PV coupling issue, a sparsity issue, or a fundamental limitation? This diagnosis requires understanding your specific system's dynamics, which only you have.
>
> **🤝 COLLAB:** If theta-gamma coupling works, derive the theoretical capacity of this working memory system together with AI. The Lisman-Idiart model predicts capacity = theta/gamma for idealised oscillators. I can help you extend this to your noisy, self-organising graph --- the math involves stochastic process theory and phase noise analysis. We can derive an analytical bound on capacity as a function of noise level, coupling strength, and network size.
>
> **VALIDATION:** Clear gamma peak in power spectrum, nested within theta. Phase-amplitude coupling is statistically significant. Multiple items segregate to different gamma phases. PC still works with oscillations active.
>
> **KILL CRITERION:** If oscillations and PC are fundamentally incompatible at all parameter settings, try: (a) alternating PC and oscillatory updates rather than running simultaneously, (b) restricting oscillations to specific subpopulations, (c) using beta rather than gamma. If none work, separate temporal computation into a distinct subgraph communicating through a hub.

**Phase 2 work split:** \~30% AI solo, \~30% collaborative, \~40% you solo.

Phase 3: Multi-System Integration Smoke Test (Month 11--15)

*Third existential test. Minimal subgraphs, maximum learning.*

3.1 Building the Minimal Subgraphs

> **🤖 AI SOLO:** Implement minimal hippocampal subgraph: \~500 recurrent nodes with fast Hebbian plasticity (high learning rate STDP). Store/recall API. Pattern completion test. This is standard attractor network implementation. I give you the Hopfield-like equations; Claude Code builds it.
>
> **🤖 AI SOLO:** Implement minimal BG subgraph: Go/No-Go competition (\~200 nodes), dopamine-modulated plasticity, binary output gate. Standard actor-critic RL. I give you the Frank 2005 equations; Claude Code builds it.
>
> **🤖 AI SOLO:** Implement minimal cerebellar subgraph: \~300 node feedforward network with supervised learning (gradient descent on prediction error). Standard regression network. Claude Code.
>
> **🤝 COLLAB:** Design the inter-subgraph connectivity. Which cortical nodes project to striatum? Which project to hippocampus? How does the thalamic relay work? I propose connectivity based on known neuroanatomy (Alexander et al. 1986, Haber 2003). You decide how to simplify it for minimal subgraphs.

3.2 The Integration Experiment

> **🤝 COLLAB:** Design the integration test task. It needs to require all three systems: cortical inference (predict sensory input), hippocampal memory (remember specific episodes), BG decision (choose actions for reward), cerebellar prediction (predict consequences of actions). I propose: a small grid-world navigation task with landmarks, reward locations, and state-dependent transitions. The system must learn the world model (cortex), remember specific locations (hippocampus), choose directions (BG), and predict the next observation (cerebellum). You decide the task complexity and success metrics.
>
> **🧠 YOU SOLO:** Run the integrated system. This is the scariest experiment in the project. You are watching four learning algorithms interact on a shared substrate in real time. Things WILL go wrong. The question is whether they go wrong in fixable ways or in fundamental ways. Watch the learning curves. Watch the weight dynamics. Watch for signs of interference: does cortical representation quality degrade when BG learning kicks in? Does hippocampal replay corrupt ongoing inference? Does cerebellar learning destabilise the cortical predictions? Your job is diagnosis. This requires the deepest understanding of your system.
>
> **UNKNOWN:** Whether multiple learning algorithms can coexist on interconnected subgraphs without catastrophic interference. This is the integration question. Original research. Nobody has done this.

3.3 Diagnosing Integration Failures

> **🤝 COLLAB:** If the system is unstable, I help you design ablation experiments: disable one subgraph at a time and measure the effect on the others. Run cortex + BG without hippocampus. Run cortex + hippocampus without BG. Find which pairwise interaction is the source of instability.
>
> **🧠 YOU SOLO:** The judgment calls: Is the instability fundamental or parametric? If fundamental, which fallback (segregated graphs, temporal segregation, asynchronous updates) to try? If parametric, which parameters to adjust? These decisions require understanding the dynamics of all four subgraphs simultaneously. This is the hardest intellectual task in the project.
>
> **🤝 COLLAB:** If integration works, work with AI to characterise the stability conditions. I can help you set up the Lyapunov analysis, compute the basin of stability numerically, and identify which parameter combinations maintain stable multi-system learning. The computational tools (eigenvalue tracking, bifurcation diagrams, sensitivity analysis) are standard --- I help you apply them to your system.
>
> **VALIDATION:** Combined system solves the navigation task faster than any single subsystem alone. One-shot adaptation to reward location changes (hippocampal contribution). Cerebellar prediction error decreases over training.
>
> **KILL CRITERION:** If no combination of parameters, update schedules, or gating mechanisms produces stable multi-system learning, see Failure Mode 4 (segregated graphs fallback).

**Phase 3 work split:** \~30% AI solo, \~30% collaborative, \~40% you solo.

**All existential risks resolved at end of Phase 3. Maximum sunk cost: 15 months.**

Phases 4--8: Refinement (Month 16+)

*The existential risks are behind you. Now build it properly.*

Phases 4--8 build each subgraph to full fidelity (full hippocampal subgraph, full BG + neuromodulation, full cerebellum, remaining mechanisms, scaling). The work allocation for these later phases follows a consistent pattern:

General Work Allocation for Phases 4--8

Implementation of Known Mechanisms

> **🤖 AI SOLO:** Any mechanism with published equations gets implemented by Claude Code. BTSP plasticity rules, BCM metaplasticity, retrograde endocannabinoid signalling, astrocytic integration, structural plasticity (edge creation/pruning). I provide the mathematical specifications from the literature; Claude Code translates to JAX. You review and test. This covers roughly 40% of each phase.

Parameter Tuning and System Integration

> **🤝 COLLAB:** Each new mechanism requires re-tuning existing parameters to maintain stability. I help you reason about expected interactions based on theory. You run the experiments and report results. We iterate. This covers roughly 20% of each phase.

Running Experiments and Building Intuition

> **🧠 YOU SOLO:** Every time a new mechanism is added, you need to observe its effects on the full system dynamics. Does adding astrocytic modulation change the oscillatory regime? Does metaplasticity stabilise or destabilise the STDP-PC interaction? Does structural plasticity prune the right edges? These questions require watching your specific system, which only you can do. This covers roughly 25% of each phase.

Theoretical Analysis

> **🤝 COLLAB:** As the system grows, the theoretical questions get harder. Work with AI on: effective dimensionality of the representational space, phase transitions as you add components, minimal subset identification. I can set up the analytical frameworks (random matrix theory for weight spectra, information geometry for representation analysis, bifurcation theory for phase transitions) and help you apply them. \~10% of each phase.

Appendix A: Timeline and Risk Summary

  ----------- ----------------------------------- -------------- ------------ ----------------------- -------------------
  **Phase**   **Core Deliverable**                **Duration**   **Cumul.**   **Existential Risk?**   **Max Sunk Cost**

  0           Graph substrate                     2 mo           Month 2      No                      2 months

  1A+1B       PC hand-built + energy experiment   4 mo           Month 6      YES --- Q1 and Q2       6 months

  2           Oscillatory dynamics                4 mo           Month 10     Moderate                10 months

  3           Multi-system integration test       5 mo           Month 15     YES --- Q3              15 months

  4           Full hippocampal subgraph           6 mo           Month 21     No                      ---

  5           Full BG + neuromodulation           7 mo           Month 28     No                      ---

  6           Full cerebellar subgraph            4 mo           Month 32     No                      ---

  7           Remaining mechanisms                6 mo           Month 38     No                      ---

  8           Scaling + benchmarks                Open           Month 39+    No                      ---
  ----------- ----------------------------------- -------------- ------------ ----------------------- -------------------

**First existential test (Q1 + Q2):** Month 6.

**Third existential test (Q3):** Month 15.

**All existential risks resolved:** Month 15. Everything after this is refinement.

**Maximum sunk cost before project-level kill:** 15 months.

Appendix B: Failure Modes and Fallbacks

*What to do when the graph tells you to do one. Each failure has a distinct signature, cause, and recovery path. Diagnosis matters --- the wrong fix for the wrong failure wastes months.*

Failure Mode 1: Topology Can't Support Signal Flow

**Signature:** PC partially works in small hand-wired subgraphs but fails on the self-organised graph. Error signals dissipate before reaching higher levels. Predictions never reach the right error nodes. Information flow analysis shows chaotic, non-hierarchical routing.

**Root cause:** The self-organising topology is too random, sparse, or tangled for reliable bidirectional hierarchical communication.

**Diagnostic:** Hand-wire a clean hierarchical topology on the same substrate (same node model, same cell types, same STDP). If PC works on the hand-wired version, the problem is self-organisation, not the substrate.

**Fallback: Constrained self-organisation.** Keep the graph but impose constraints: fixed hierarchical levels, fixed edge type ratios between levels. Self-organisation operates only within levels (lateral connectivity, feature selectivity). You lose the emergent hierarchy claim but keep everything else.

**What you keep:** \~80% of the project.

**What you lose:** The self-organising hierarchy story.

Failure Mode 2: Node Model Too Weak

**Signature:** The network learns something but representations are weak, noisy, and low-dimensional. Apical gating has negligible effect. Increasing network size doesn't help.

**Root cause:** Two compartments don't capture enough dendritic computational power. The gating is too blunt.

**Diagnostic:** Replace the two-compartment model with a multi-layer perceptron at each node (3--5 hidden units as dendritic branches). If this dramatically improves representations, the problem is node power.

**Fallback: Multi-compartment nodes with quadratic integration.** Expand to 3--5 compartments per excitatory node, each performing quadratic integration: output = xᵀAx + w·x + b per compartment. Directly from the NeurIPS 2024 dendritic paper. Cost: \~3--5x more parameters per node.

**What you keep:** Everything except the specific node model.

**What you lose:** Simplicity. Multi-compartment nodes are harder to debug and have more parameters.

Failure Mode 3: STDP Fights Predictive Coding

**Signature:** PC initially works weakly but degrades over time. STDP strengthens edges that increase prediction error (because large errors create strong postsynaptic activity, triggering LTP). The system learns to be more surprised, not less. Turning off STDP stabilises PC.

**Root cause:** STDP's causal learning rule is misaligned with PC's objective. Without a third gating factor, STDP strengthens whatever causes firing, including pathological error-driven firing.

**Diagnostic:** Freeze all weights. Run PC with fixed random weights. If inference still converges, the architecture supports PC and the problem is purely the learning rule.

**Fallback A: Three-factor STDP.** Gate STDP with the sign of the local prediction error. LTP only triggers when prediction error is decreasing. Biologically plausible --- neuromodulatory gating of plasticity is well-documented.

**Fallback B: PC-native updates.** Use Friston's variational free energy gradient directly. Weight update = prediction_error × presynaptic_activity. Local and Hebbian-like but explicitly aligned with error minimisation. Less exciting, works.

**Fallback C: Gradient diagnosis.** Use JAX autodiff to backpropagate prediction error through the graph. Compare the exact gradient to what STDP is doing. If anti-correlated, you know exactly which edges need a different rule. Not a production solution but a powerful diagnostic.

**What you keep:** The entire architecture. All subgraphs. All mechanisms except STDP itself.

**What you lose:** The elegance of a single generic learning rule producing everything. Most likely failure mode, cheapest to fix.

Failure Mode 4: Multi-System Integration Impossible

**Signature:** Individual subgraphs work in isolation. Connecting any two degrades both. Cortical representations drift when BG acts. Hippocampal replay corrupts cortical inference. No combination of learning rates or gating resolves it.

**Root cause:** Different learning algorithms are fundamentally incompatible on a shared substrate. Each assumes the rest of the system is stable while it learns. When all learn simultaneously, none have a stable target.

**Diagnostic:** Train systems sequentially, not simultaneously. If sequential training works, the problem is concurrent learning, not architectural incompatibility.

**Fallback A: Segregated graphs.** Build each subgraph as an independent graph. Connect through a small set of interface nodes (thalamic hub). Each subgraph runs its own learning algorithm on its own topology. Less elegant, might be necessary. The brain IS like this at a coarse scale.

**Fallback B: Temporal segregation.** Keep single graph but run different algorithms at different times. "Wake" = cortical PC + BG RL. "Sleep" = hippocampal replay + consolidation. "Practice" = cerebellar learning. Biologically realistic --- different brain states enable different learning modes.

**What you keep:** All individual subgraph designs. All learning algorithms.

**What you lose:** The unified single-graph vision. System becomes a federation of specialised subgraphs.

Failure Mode 5: It Works But Nobody Cares (Including You)

**Signature:** Architecture functions correctly. PC works. Memory works. Action selection works. But on every task you try, a simpler approach matches or exceeds performance. The neuromorphic graph offers no measurable advantage for anything you actually want to do with it.

**Root cause:** You're testing on the wrong tasks. For tasks with abundant data and available compute, transformers are hard to beat. Your architecture's advantages --- efficiency, few-shot learning, continual learning --- only show on tasks that need those properties.

**Tasks where your architecture SHOULD win:** One-shot learning from limited data. Continual learning without catastrophic forgetting. Energy-efficient inference (measure FLOPs per prediction, not wall-clock on GPU). Temporal sequence prediction with long-range structure. Adapting to non-stationary environments. Anomaly detection from native prediction error signals.

**Tasks where transformers always win:** Large-scale supervised classification with millions of labelled examples. Anything where abundant compute and data are available and efficiency doesn't matter.

**Fallback: Distillation.** Use the neuromorphic graph as a teacher to train simpler architectures. Run your system on tasks, record its behaviour, train a conventional network to mimic it. The graph becomes a research tool for discovering computational strategies; the deployed tool is a simpler system that inherited those strategies.

**The real competitive angle:** You don't need to beat transformers on accuracy. You need to beat them on accuracy-per-watt. The brain runs on 20 watts. If your architecture achieves 10% of a transformer's capability at 1% of the energy, that's a viable tool for edge computing, embedded systems, and anywhere you can't afford a data centre.

Appendix C: Work Allocation Summary

  ----------------- --------------------------------- ------------------------------------------ -------------------------------------- ------------------
  **Phase**         **AI Solo**                       **Collaborative**                          **You Solo**                           **Total Months**

  0: Substrate      60% --- Infra, code, viz          25% --- Node model design                  15% --- Parameter intuition            2

  1A: PC            35% --- PC code, analysis         30% --- STDP-PC design                     35% --- Experiments, judgment          4

  1B: Energy        25% --- Energy impl, sweeps       30% --- Functional design, math analysis   45% --- Observing emergence, insight   (parallel)

  2: Oscillations   30% --- Gap junctions, spectral   30% --- Theta design, capacity math        40% --- Oscillation-PC interaction     4

  3: Integration    30% --- Minimal subgraphs         30% --- Task design, stability analysis    40% --- Diagnosing failures            5

  4-8: Refinement   40% --- Mechanism code            25% --- Tuning + theory                    35% --- Observation + decisions        22+
  ----------------- --------------------------------- ------------------------------------------ -------------------------------------- ------------------

Overall Project Split

**AI Solo:** \~35% of total work. All infrastructure, implementation of known mechanisms, analysis pipelines, parameter sweep setup, and mathematical frameworks (stability analysis, information theory, spectral analysis).

**Collaborative (You + AI):** \~28% of total work. Experiment design, architecture decisions, debugging, interpretation, mathematical derivations applied to your system, formalising results.

**You Solo:** \~37% of total work. Running experiments, observing dynamics, building intuition, making judgment calls, kill/pivot/continue decisions, and the creative leaps --- noticing patterns nobody has described before.

A third of this project is stuff you never have to think about. Use that ruthlessly. Spend your freed-up cycles on the 37% that only you can do --- that's where the value lives. The system either works or it doesn't. Nobody needs to know about it but you.
