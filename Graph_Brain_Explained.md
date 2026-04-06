# Graph Brain: How It Works

## The Big Idea

Imagine a city where every building can talk to its neighbours. Messages travel along roads. Some roads carry news ("here's what's happening"), others carry predictions ("here's what I think will happen"). Every building listens to both, and if the prediction matches the news, the building stays quiet — no need to shout about something everyone already expected. But if something unexpected happens, that building lights up and tells everyone.

Now imagine the city builds itself. Nobody designs the road layout. Nobody assigns which buildings listen to which others. The roads grow where they're needed and disappear where they're not. Over time, the city organises itself into a system that's very good at predicting what happens next — and very efficient, because it only uses energy to communicate surprises.

That's Graph Brain. Except the buildings are artificial neurons, the roads are connections between them, and the predictions are about patterns in data rather than city gossip.

---

## What Is a Graph?

A graph is just dots connected by lines. In maths, the dots are called "nodes" and the lines are called "edges." Your social network is a graph — you're a node, your friends are nodes, and the friendships are edges.

Graph Brain is a graph where:
- **Nodes** are artificial neurons (about 1,000-5,000 of them)
- **Edges** are connections that carry signals between neurons
- The whole thing runs on a GPU (the same chip that renders video games) and processes thousands of signals simultaneously

What makes it different from a normal neural network (like ChatGPT or image generators): the graph builds itself. The connections aren't designed by an engineer — they emerge from the system trying to be energy-efficient while also understanding its input.

---

## The Core Principle: Prediction Saves Energy

The brain uses about 20 watts — the same as a dim light bulb. It achieves this despite processing an enormous amount of information by using a trick: **don't transmit what you already expect.**

If you're sitting in your living room and nothing changes, your visual system goes quiet. It predicted "same room" and the prediction was correct, so there's nothing to report. But if a spider drops from the ceiling, your visual system SCREAMS — that wasn't predicted.

Graph Brain works the same way. Each neuron has two inputs:
- **Evidence** (what's actually happening) — arrives from below
- **Prediction** (what was expected) — arrives from above

The neuron computes the DIFFERENCE. If evidence matches prediction, the neuron is silent. If they don't match, it fires loudly. This is called **predictive coding** — the system only communicates prediction errors, not the full signal.

The energy rule: activity costs energy. Being silent is free. The cheapest way to be silent is to predict correctly. So the system is MOTIVATED to get better at predicting — not because we told it to, but because good prediction literally costs less energy.

---

## How It Learns

### The Hebbian Rule: Neurons That Fire Together Wire Together

The oldest learning rule in neuroscience. If two neurons are active at the same time, the connection between them gets stronger. If they're never active together, the connection weakens and eventually disappears.

In Graph Brain, this means the system naturally discovers which neurons are related — which ones tend to be active at the same time. Over many repetitions of a pattern, the connections that carry useful predictions get stronger, and useless connections fade away.

### The Energy Constraint: Be Efficient or Be Punished

Every active neuron pays an energy cost. Every connection has a maintenance cost. The system is constantly pressured to use as few neurons and connections as possible.

This creates a tension: the system WANTS to be silent (save energy) but CAN'T be silent unless it predicts correctly (because unexpected input forces neurons to fire). The only way to satisfy both demands simultaneously is to build accurate predictions.

This tension — between energy efficiency and prediction accuracy — is what drives the entire self-organisation process. Nobody tells the system what to learn. The physics of the energy constraint makes learning the only viable strategy.

---

## What Makes This Different

### It Builds Itself

Most AI systems have their architecture designed by engineers. The number of layers, the connectivity pattern, which neurons do what — all decided in advance.

Graph Brain starts as a random collection of neurons with random connections. The energy constraint and learning rules do the rest. The system discovers its own architecture — which neurons become prediction generators, which become error detectors, how information flows through the network. We don't assign roles. They emerge.

### Multiple Learning Systems Coexist

The brain doesn't have just one learning mechanism. It has at least five, all running simultaneously on the same tissue:
- **Prediction** (cortex): building a model of the world
- **Reward learning** (basal ganglia): learning what actions lead to good outcomes
- **Memory** (hippocampus): remembering specific experiences
- **Timing** (cerebellum): learning precise sequences
- **Emotional learning** (amygdala): associating stimuli with good or bad feelings

Graph Brain implements the first three on the same graph. The prediction system, the reward system, and the memory system all share the same neurons and connections. They don't interfere with each other — they cooperate. The prediction system tells the reward system what to expect. The reward system tells the prediction system what's worth predicting. The memory system gives both of them access to past experience.

### It Runs on Standard Hardware

No special neuromorphic chips. No custom silicon. A standard gaming GPU (the kind used for playing video games) runs a 5,000-neuron Graph Brain at over 100 steps per second. The architecture is designed to scale — the same principles work whether you have 5,000 neurons or 5 million.

---

## The Architecture in Pictures

### The Node (Neuron)

Each neuron has two compartments, like a house with two floors:

```
    ┌─────────────┐
    │   APICAL     │  ← Predictions arrive here (from above)
    │  (upstairs)  │     "Here's what I think will happen"
    ├─────────────┤
    │   BASAL      │  ← Evidence arrives here (from below)
    │ (downstairs) │     "Here's what's actually happening"
    └──────┬──────┘
           │
     OUTPUT = |upstairs - downstairs|

     Match → quiet (energy saved)
     Mismatch → loud (error signal)
```

### The Edge Types (Roads)

Six types of connections, each carrying a different kind of signal:

```
  DRIVING ──────→  Carries evidence/errors UPWARD
                   "Something unexpected happened down here"

  MODULATORY ───→  Carries predictions DOWNWARD
                   "Based on what I know, here's what to expect"

  PV INHIBITION ─→  Controls the volume
                    "Everyone quiet down" (winner-take-all competition)

  SST INHIBITION ─→  Gates the prediction channel
                    "Ignore the prediction for now"

  VIP ───────────→  Releases the gate
                    "Wait, pay attention to the prediction again!"

  GAP JUNCTIONS ──→  Synchronises timing between inhibitory neurons
                    "Let's all fire together" (creates brain rhythms)
```

### The Self-Organisation Process

```
  START: Random connections, no structure
    │
    │  Energy pressure: "be quiet!"
    │  Input arrives: "but I can't be quiet if I'm surprised!"
    │  Only solution: "predict correctly, then I can be quiet"
    │
    ▼
  MIDDLE: Connections strengthen where predictions work
    │
    │  Successful predictions → connections consolidate
    │  Failed predictions → connections weaken
    │  Unused connections → pruned (saves energy)
    │
    ▼
  RESULT: Self-organised prediction hierarchy

    Upper neurons predict what lower neurons will see
    Lower neurons only report surprises
    Energy usage drops as predictions improve
    System detects unexpected events (mismatch detection)
```

---

## What We've Proven

In five days of development and experimentation:

**The system learns to predict.** Given alternating patterns (A then B then A then B...), the system learns that A predicts B and B predicts A. When we violate this expectation (presenting A then A instead of A then B), the system's error neurons fire 37% harder than normal. It detected the violation — without anyone telling it what to look for.

**It does this entirely by itself.** No hand-designed hierarchy. No assigned neuron roles. No pre-specified connectivity. The energy constraint and learning rules produce prediction-capable structure from a random starting point.

**Multiple systems coexist.** When we added a reward-learning system to the prediction system, neither destroyed the other. The prediction system kept predicting (neuron health stayed stable). The reward system learned which actions to take (75% accuracy on a pattern-matching task). They cooperated on the same graph.

**It scales.** The same result reproduces across different random starting configurations (5 out of 5 tested) and at 4x larger graph size. The principles work regardless of specific initial conditions.

---

## Why This Matters

Most AI today (large language models, image generators) works by processing data through a fixed pipeline designed by engineers. They're powerful but rigid — they can't reorganise their own architecture, they can't learn new things without forgetting old things, and they consume enormous amounts of energy.

The brain does all three: it reorganises constantly, it learns without catastrophic forgetting (mostly), and it runs on 20 watts. Graph Brain is an attempt to capture the principles that make this possible — not by simulating the brain's biology, but by implementing the computational principles the brain uses:

- **Predict, don't transmit** (energy efficiency through prediction)
- **Let structure emerge** (self-organisation through energy constraints)
- **Multiple systems, one substrate** (cooperation through shared representation)

If these principles scale — and the early evidence suggests they do — this architecture could be the foundation for AI systems that learn continuously, adapt their own structure, and operate at a fraction of the energy cost of current approaches.

---

## Glossary

**Node/Neuron**: A computational unit that receives input, processes it, and produces output. In Graph Brain, each has two input compartments (basal and apical).

**Edge/Connection**: A link between two neurons that carries a signal. Has a weight (strength) and a delay (travel time).

**Predictive Coding**: The strategy of only communicating prediction errors, not the full signal. Saves energy because correct predictions produce silence.

**Hebbian Learning**: "Neurons that fire together wire together." Connections strengthen between co-active neurons.

**Energy Constraint**: The rule that activity and connections cost energy. Forces the system to be efficient, which drives self-organisation.

**Mismatch Detection**: The system's ability to detect when something unexpected happens — the error neurons fire harder for violations than for expected input.

**Self-Organisation**: Structure emerging from simple rules rather than being designed. Nobody tells the neurons what to do — the energy constraint and learning rules produce useful computation automatically.

**Sparsity**: Most neurons being silent most of the time. The system activates only the neurons it needs, like a city where most buildings are dark and only a few are lit up.
