# Hybrid SAM + ViT-DAVE2 Architecture Review

## Executive Verdict

Turning this project into a hybrid perception plus end-to-end driving system is a good idea.

Turning SAM into the whole perception stack is not.

SAM is strong as a dense scene prior, especially for road masks, lane support masks, and object extent proposals. It is weak as a standalone detector, weak as a tracker, and not calibrated for control-critical geometry without extra structure. A publishable system should use SAM-derived scene understanding, then convert it into explicit road, lane, object, and risk signals before fusion with the policy model.

The right architecture here is:

- ViT-DAVE2 for policy-side visual encoding
- SAM-style multi-scale encoder for dense scene understanding
- explicit road and lane geometry heads
- query-based object decoder plus tracker
- structured control feature extractor
- multi-branch fusion with cross-attention and control-feature gating

That is what the code implements.

## Phase 1: System Design

### Current Project Reality

The current codebase is still a classic behavioral cloning pipeline:

- single RGB frame
- crop, resize, YUV normalization
- ViT encoder
- scalar steering regression head
- no temporal memory
- no explicit perception state
- no road model
- no lane representation
- no object reasoning

That architecture is too brittle for research claims about robustness. It can fit imitation data, but it cannot explain failures, cannot localize hazard sources, and cannot maintain stable control under distribution shift.

### Recommended Research Architecture

```text
src/dave2/
├── model.py
│   ├── DAVE2VisionEncoder
│   └── DAVE2
├── perception.py
│   ├── SAMBackboneAdapter
│   ├── FeaturePyramidAggregator
│   ├── SceneTokenizer
│   ├── ObjectDetectionHead
│   ├── LaneDetectionModule
│   ├── TrackingModule
│   ├── ControlFeatureExtractor
│   └── SAMPerceptionWrapper
├── fusion.py
│   └── HybridFusionModule
├── hybrid_model.py
│   ├── DrivingPolicyHead
│   └── HybridDrivingModel
└── config.py
    ├── PerceptionConfig
    ├── FusionConfig
    ├── PolicyConfig
    └── HybridModelConfig
```

### Module Responsibilities

`DAVE2VisionEncoder`

- encodes raw driving images into policy-oriented visual tokens
- preserves appearance, texture, and scene layout cues that imitation models use well
- should not be responsible for explicit road geometry by itself

`SAMPerceptionWrapper`

- produces multi-scale scene features and compact scene tokens
- decodes dense semantic segmentation
- decodes explicit binary drivable road mask
- decodes lane support masks plus anchor-based lane boundaries
- predicts object boxes and classes
- tracks objects across frames
- converts perception outputs into control-relevant features

`TrackingModule`

- assigns persistent IDs
- estimates 2D motion from repeated detections
- supports TTC proxy and proximity reasoning

`ControlFeatureExtractor`

- turns perception into signals the controller can actually use:
- lane center offset
- heading deviation
- lane width
- lane visibility
- lane curvature proxy
- drivable area ratio
- drivable center offset
- drivable confidence
- object proximity risk
- object lateral offset
- monocular TTC proxy

`HybridFusionModule`

- fuses image tokens, scene tokens, and control features
- exposes ablation strategies without rewriting the model

`DrivingPolicyHead`

- predicts steering, throttle, brake
- optionally predicts action uncertainty

### Perception-to-Control Information Flow

```text
RGB image
  -> ViT-DAVE2 encoder
      -> visual policy tokens

RGB image
  -> SAM-style encoder + FPN
      -> dense scene feature map
      -> scene tokens
      -> semantic segmentation
      -> road mask
      -> lane boundaries + visibility anchors
      -> object queries
      -> tracker state
      -> control feature vector

visual policy tokens + scene tokens + control feature vector
  -> multi-branch fusion
  -> driving policy head
  -> steering, throttle, brake, uncertainty
```

### Fusion Strategy Comparison

#### Early Fusion

What it does:

- injects perception features into the visual stream before high-level reasoning is complete

Why it is attractive:

- allows low-level interactions between raw image and perception priors

Why it is weak here:

- brittle when SAM masks are noisy
- couples policy encoder too tightly to perception errors
- difficult to stabilize in low-data training
- expensive to retrain when perception changes

Reviewer verdict:

- fine for ablations
- not the best default research choice

#### Late Fusion

What it does:

- encodes perception and policy separately, combines only near the final head

Why it is attractive:

- stable
- easy to train
- robust to component swaps

Why it is weak here:

- throws away rich token-level scene context
- perception only acts as a coarse correction term
- underuses road and lane structure

Reviewer verdict:

- strong baseline
- not enough if the claim is deep perception-policy integration

#### Cross-Attention Fusion

What it does:

- lets policy tokens query perception tokens

Why it is attractive:

- interpretable
- targeted information exchange
- better than crude concatenation

Why it is weak here:

- if used alone, it still ignores some structured geometry
- attention can drift toward salient but control-irrelevant objects

Reviewer verdict:

- good component
- not sufficient as the whole design

#### Multi-Branch Fusion

What it does:

- keeps three branches:
- policy visual tokens
- dense scene tokens
- structured control features
- uses cross-attention plus gated aggregation

Why it is strong:

- preserves dense scene understanding
- preserves explicit geometry
- keeps ablation-friendly separation
- less fragile than forcing all knowledge into one token space

Reviewer verdict:

- best practical choice for this project

#### Token-Level Fusion

What it does:

- concatenates all modality tokens into one transformer

Why it is attractive:

- elegant
- expressive
- strong on paper when data and compute are abundant

Why it is weak here:

- expensive
- data-hungry
- easy to overfit
- difficult to debug
- often overkill unless you already have large-scale multimodal driving data

Reviewer verdict:

- publishable only if you actually have enough data and compute
- otherwise it is mostly hype

### Recommended Fusion Strategy

Use `multi_branch` as the default.

Why:

- it is the best balance between expressiveness, robustness, and debuggability
- it preserves road and lane geometry as first-class control inputs
- it allows the policy to attend to dense scene tokens without depending entirely on them
- it is easier to analyze when perception fails
- it is realistic for CARLA plus low-data real-world transfer

If this were submitted to CVPR or ICRA, the strongest claim is not that the network is end-to-end. The strongest claim is that the network is end-to-end while keeping explicit road understanding and uncertainty-aware control hooks.

## Phase 2: SAM Extension Design

### A) Object Detection plus Tracking

Bad idea:

- using raw SAM masks as your detector

Better idea:

- use SAM features and masks as object extent priors
- run a query-based detection head on SAM scene tokens
- attach a tracker using box overlap plus embedding similarity

Why:

- SAM can segment prominent objects but it is not a robust road-scene detector by itself
- autonomous driving needs stable categories and temporal IDs, not only masks

### B) Image Segmentation

Use SAM as the dense segmentation backbone.

This is the natural fit:

- scene classes
- object silhouettes
- road context
- curb-like structure priors

This module should remain auxiliary to control, not the direct control input.

### C) Road Detection

Road detection must be a first-class head.

Representation:

- binary drivable area probability map
- confidence score over lower and middle image regions
- drivable centerline estimate from mask centroid

Why not bury road inside generic segmentation:

- road is the dominant control prior
- you want a direct training signal
- you want a direct safety monitor
- you want the policy to know when drivable support is weak

### D) Lane Detection and Lane Tracking

Lane boundaries should be represented as:

- dense boundary logits for left and right lane support
- fixed vertical anchor samples with:
- left boundary x
- right boundary x
- left visibility
- right visibility

This is better than only a lane mask because control depends on geometry, not merely existence.

Lane tracking should be:

- temporal smoothing of anchor predictions
- ID consistency for lane instances if multi-lane topology is needed
- confidence decay when weather or occlusion corrupts observations

### E) Lane Keeping Signals for Driving Control

The controller should not consume raw masks directly.

It should consume compact signals such as:

- lane center offset
- heading deviation
- lane width
- lane curvature proxy
- lane visibility confidence
- drivable area ratio
- drivable center offset
- drivable confidence
- object proximity risk
- TTC proxy

### Lane Boundary Representation

Use normalized anchor-wise boundaries.

For anchor row `i`:

- `left_x[i] in [0, 1]`
- `right_x[i] in [0, 1]`
- `visibility[i] in [0, 1]^2`
- `anchor_y[i]` fixed from ego-near to horizon

This gives:

- direct centerline geometry
- easy curvature estimation
- easy control-feature conversion
- good supervision in both CARLA and real datasets

### Drivable Area Representation

Use:

- a dense drivable mask
- a scalar drivable area ratio
- a scalar drivable center offset
- a scalar drivable confidence

This is essential because lane marks fail in:

- construction zones
- faded paint
- snow
- night glare
- rural roads

Road understanding must survive when lane cues disappear.

### Converting Perception to Control-Relevant Features

`lane_center_offset`

- difference between estimated lane center and ego image center

`heading_deviation`

- difference between near-field and far-field lane center

`drivable_area_confidence`

- mean road probability in the forward path region

`object_proximity_risk`

- objectness x area x near-center weighting

`TTC`

- use tracker motion and box growth to derive a monocular TTC proxy
- be honest in the paper: it is a proxy, not metric depth TTC

## Phase 3: Implementation Notes

The code now contains:

- `SAMPerceptionWrapper`
- `LaneDetectionModule`
- `TrackingModule`
- `HybridFusionModule`
- `DrivingPolicyHead`
- `HybridDrivingModel`

Important engineering choices:

- the SAM wrapper accepts an external backbone but does not hard-code a third-party SAM package
- this keeps the project pure PyTorch and avoids fragile dependency lock-in
- the fallback encoder is an interface-compatible stand-in, not a claim that the fallback is real SAM

This is the correct engineering choice because claiming a real SAM stack without shipping the actual backbone integration would be misleading.

## Phase 4: Data Pipeline

### Recommended Dataset Structure

```text
dataset_root/
├── scenes/
│   ├── sequence_000001/
│   │   ├── rgb/
│   │   ├── semantic/
│   │   ├── road_mask/
│   │   ├── lane_mask/
│   │   ├── lane_anchors/
│   │   ├── detections/
│   │   ├── tracking/
│   │   ├── ego/
│   │   └── manifest.jsonl
└── splits/
    ├── train.json
    ├── val.json
    └── test.json
```

Each frame record should contain:

- image path
- timestamp
- steering
- throttle
- brake
- speed
- yaw rate
- route command if available
- road mask path
- lane mask path
- lane anchor label path
- semantic mask path
- object boxes and classes
- object track IDs
- weather and domain tags

### Labels Needed

- control actions
- binary road mask
- lane boundaries or anchor geometry
- semantic segmentation
- object boxes and classes
- track IDs
- optional route command
- optional future trajectory for planner-aware supervision

### Annotation Strategy

Best practical strategy:

- generate dense labels in CARLA automatically
- pretrain perception there
- use SAM-assisted pseudo-labeling on real data
- manually correct only high-value subsets:
- intersections
- merges
- night
- rain
- occlusion-heavy scenes

Do not fully hand-label everything. That is expensive and wasteful.

### Simulator Support

CARLA is the right simulator here because it gives:

- semantic classes
- lane topology
- drivable map
- actor IDs
- weather variation
- route commands
- future state and trajectory labels

### Real-World Compatibility

You need a schema that survives both simulator and real logs.

That means:

- normalized boxes
- normalized lane anchors
- binary road masks
- control labels in the same units
- optional missing-label handling for partial supervision

### Training Strategy

#### Perception Modules

Train first with supervised multi-task learning:

- semantic loss
- road BCE or Dice
- lane mask plus anchor regression
- detection loss
- tracking consistency loss

#### Control Policy

Train second with imitation learning:

- steering
- throttle
- brake
- optional uncertainty-weighted regression

#### Joint Fine-Tuning

Then fine-tune jointly:

- freeze lower perception layers at first
- unfreeze progressively
- keep auxiliary perception losses active
- do not let control loss erase road understanding

### Learning Paradigm Comparison

Supervised learning:

- strongest for perception
- necessary

Imitation learning:

- best default for low-data end-to-end driving
- stable
- efficient

Offline RL:

- usually overrated in low-data autonomous driving
- reward misspecification is brutal
- dataset coverage is usually inadequate

Diffusion policy:

- useful when action multimodality matters
- overkill for a first hybrid system unless you already have strong temporal data

### Best Low-Data Regime Strategy

Recommended:

- CARLA multi-task pretraining for road, lane, segmentation, and detection
- real-world imitation learning with frozen or partially frozen perception
- confidence-aware joint fine-tuning on a small curated real subset

Do not start with offline RL.
Do not start with a diffusion policy.
Do not try to learn road understanding purely from actions.

## Phase 5: Failure Analysis

### Likely Failure Modes

1. Perception-policy disagreement

- SAM branch says road is safe
- policy branch overreacts to texture
- control becomes oscillatory

Fix:

- confidence-aware gating
- disagreement regularization
- intervention trigger when branches diverge

2. Road-mask overconfidence

- wet roads
- snow
- night glare
- unusual pavement colors

Fix:

- uncertainty estimation on road head
- explicit out-of-domain augmentation
- confidence thresholding before control trust

3. Lane collapse under adverse conditions

- faded paint
- occlusion
- shadow
- rain

Fix:

- road-first fallback
- temporal smoothing
- lane-visibility-aware control weighting

4. Tracking instability

- object IDs switch
- TTC becomes noisy

Fix:

- stronger appearance embeddings
- motion prior
- track age and miss logic
- camera-motion compensation if ego motion is known

5. Latency bottleneck

- SAM-style dense perception plus ViT policy plus tracker can become too slow

Fix:

- share image encoder where possible
- compress scene tokens
- update heavy perception at lower frequency than control head
- cache track memory

6. Domain gap

- CARLA roads are cleaner than reality
- control policy overfits simulator lane geometry

Fix:

- aggressive domain randomization
- style and weather variation
- real-data fine-tuning with road masks and lane anchors

7. Policy shortcut learning

- model uses sky color, hood texture, or road edges

Fix:

- strong augmentations
- intervention-heavy data
- occlusion tests
- attribution auditing

### Harsh Reviewer Take

If the paper says "SAM improves autonomous driving" without:

- road-specific metrics
- lane-specific metrics
- closed-loop control results
- failure-case analysis
- latency accounting

then the claim is weak.

If the paper uses only open-loop steering MSE, it is not convincing.

## Phase 6: Research Upgrade Toward Publishable Quality

### Uncertainty Estimation

Valuable.

Not hype.

Use it for:

- road confidence
- lane confidence
- action variance
- safety gating

### Confidence-Aware Control

Valuable.

Probably mandatory.

When road and lane confidence drop, the controller should:

- reduce action magnitude
- prefer conservative throttle
- defer to route or planner priors if available

### Trajectory Prediction

Valuable if used as an auxiliary objective.

Not necessary as a first output head unless the evaluation includes planning tasks.

### Temporal Memory

Very valuable.

Probably the next real upgrade after this patch.

Single-frame control remains fragile.

Recommended next step:

- add temporal transformer memory or recurrent state over scene tokens and control features

### Multi-Frame Reasoning

Valuable.

Especially for:

- tracking
- TTC
- partial occlusion
- lane recovery

### Diffusion Action Head

Mostly hype for this project right now.

It becomes useful only if:

- you have multimodal futures
- you have enough data
- your baseline policy already saturates

Otherwise it adds complexity faster than it adds robustness.

### Planner-Aware Policy Learning

Very valuable if route commands or coarse planner trajectories are available.

Best use:

- add planner token or route token
- train policy to remain road-safe while route-conditioned

This is far more useful than diffusion in a low-data regime.

## Final Recommendation

Build the system around:

- road-first perception
- explicit lane geometry
- object and tracking side channels
- multi-branch fusion
- uncertainty-aware control
- later temporal memory

Do not sell SAM as a universal detector-tracker-planner.

Sell it as a strong dense-scene prior inside a structured autonomous driving stack.

That is the defensible research story.
