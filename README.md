# Adversarial Learning for Counterfactual Fairness

This work has been accepted at the Machine Learning Journal.

https://link.springer.com/article/10.1007/s10994-022-06206-8

Link to arxiv paper: https://arxiv.org/pdf/2008.13122.pdf

Abstract—In recent years, fairness has become an important
topic in the machine learning research community. In particular,
counterfactual fairness aims at building prediction models which
ensure fairness at the most individual level. Rather than globally
considering equity over the entire population, the idea is to
imagine what any individual would look like with a variation
of a given attribute of interest, such as a different gender
or race for instance. Existing approaches rely on Variational
Auto-encoding of individuals, using Maximum Mean Discrepancy
(MMD) penalization to limit the statistical dependence of inferred
representations with their corresponding sensitive attributes.
This enables the simulation of counterfactual samples used for
training the target fair model, the goal being to produce similar
outcomes for every alternate version of any individual. In this
work, we propose to rely on an adversarial neural learning
approach, that enables more powerful inference than with MMD
penalties, and is particularly better fitted for the continuous
setting, where values of sensitive attributes cannot be exhaustively
enumerated. Experiments show significant improvements in term
of counterfactual fairness for both the discrete and the continuous
settings.

<p align="center">
  <img src="https://github.com/fairml-research/Counterfactual_Fairness/blob/main/img.png?raw=true" width="550" title="hover text">
</p>

