# Expert Memo — Red / Variational — pifb-pullback-mechanism (opening)

## Steelman

Hierarchical active inference does separate timescales: fast state inference (beliefs $q$ over latent states) and slow parameter/structure learning (the generative model $s$) [FristonEtAl2017; Friston2017Graphical]. Mapping the four pullback tiers onto this $h\to s\to p\to q$ hierarchy (:2782) is internally coherent, and identifying the slow, quasi-static generative-model tier $G^{(s)}$ with "stable structural background" (:2793) is a defensible organizing analogy: the slow channel does encode the structural assumptions against which fast beliefs are set. The manuscript also flags its own caveat honestly (:2797), distinguishing Hoffman's species-evolutionary level from a within-lifetime Bayesian level, which is the kind of disclosure a careful reading demands.

## Strongest falsification

Two problems, both about whether the tier-identification is *load-bearing* or *decorative*.

First, the central proposal — "what an agent phenomenally experiences as spatial structure is most naturally identified with $G^{(s)}$ rather than $G^{(p)}$ or $G^{(q)}$" (:2795) — is stated with "we propose" and "most naturally," and the manuscript supplies no condition under which the identification would be *wrong*. It is a preference among coexistent tensors (:2782 calls all four "coexistent rather than alternative"), not a derived result. There is no measurement, no behavioral signature, no consequence that distinguishes "perceived space lives in $G^{(s)}$" from "perceived space lives in $G^{(p)}$." The variational machinery does not adjudicate it; the manuscript admits as much ("does not adjudicate the metaphysical question," :2797). That makes the identification interpretive, not a claim the framework establishes.

Second, the [Friston2017] citation is loaded more heavily than the source supports. The manuscript writes that the hierarchical active-inference literature "locates the structural part of perceptual organization in Bayesian parameter learning of the generative model rather than in Bayesian inference at the state level" (:2797). The active-inference literature distributes structural/perceptual organization across *deep hierarchical generative models* (nested temporal and state structure across levels), and treats parameter learning (e.g., learning $A$/$B$ matrices, Bayesian model reduction) and state inference as complementary free-energy-minimizing processes — it does not single out parameter learning as *the* carrier of "perceived space" [FristonEtAl2017, "Active inference: a process theory"; ParrPezzuloFriston2022]. The manuscript's sentence reads a specific localization ("structural perception = parameter learning") into a literature that places structure in the hierarchical depth of the model, not specifically in the slow-parameter channel. This is a wrong-strength citation: the source supports "structure is hierarchical and partly slow," not "perceived space is the parameter-learning tier."

The honest Hoffman caveat (:2797) is genuine and I credit it. But it does not rescue the $G^{(s)}$ identification, because the caveat concerns attribution to Hoffman, not the falsifiability of the proposal itself.

## External citation

[FristonEtAl2017, "Active inference: A process theory," Neural Computation 29:1–49]: structural organization in active inference is carried by hierarchical/deep generative models with nested state and temporal structure; parameter learning (slow) and state inference (fast) are complementary processes minimizing the same variational free energy — the framework does not localize "perceived spatial structure" specifically in the parameter-learning channel. [ParrPezzuloFriston2022, *Active Inference*, MIT Press]: the standard decomposition treats model structure as distributed across hierarchical levels, not as a single slow-parameter tier identified with phenomenal space.

## Falsification condition

The variational attack is wrong if the manuscript states a condition under which "perceived space $=G^{(s)}$" would be falsified (a behavioral or measurable discriminator between the tiers), or if [FristonEtAl2017] explicitly localizes perceptual spatial structure in parameter learning as opposed to hierarchical inference. Neither is present: the manuscript offers a preference with no discriminator (:2795, :2797), and the source distributes structure across the hierarchy.

## Newly-discovered canon

- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., Pezzulo, G. (2017), "Active inference: A process theory," *Neural Computation* 29:1–49: hierarchical/deep generative models; parameter learning and state inference are complementary free-energy-minimizing processes across levels. (ResearchGate record: https://www.researchgate.net/publication/310627938_Active_Inference_A_Process_Theory)
- Active-inference synthesis literature (e.g., Da Costa et al., "Active inference on discrete state-spaces: A synthesis," *J. Math. Psychol.* 2020) places structural learning in Bayesian model reduction / hierarchical depth, not in a single slow-parameter tier. URL: https://www.sciencedirect.com/science/article/pii/S0022249620300857
