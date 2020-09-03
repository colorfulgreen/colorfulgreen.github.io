---
layout: post
title:  "PRML Notes：Graphical Models"
date:   2020-09-03 09:00:00 +0800
categories: machine-learning 
---

> "A graphical model is a way to represent a joint distribution by making use of conditional independence assumptions.“

Contents

- [1 Motivation](#motivation)
- [2 Conditional Independence](#conditional-independence)
- [3 Markov Random Fields](#markov-random-fields)
    - [3.1 Conditional independence properties]
        - [3.1.1 Why do we need undirected graphical models?]
        - [3.1.2 How to do conditional independence test?]
    - [3.2 Factorization properties]
    - [3.3 Illustration: Image de-noising]

The probability can be expressed in terms of two simple equations corresponding to the sum rule and the product rule:

| sum rule     | $$ p(X) = \sum_{Y} p(X, Y) $$  |
| product rule | $$ p(X,Y) = p(Y\|X)p(X) $$     |

We could formulate and solve complicated probabilistic models purely by algebraic manipulations which amount to repeated application of these two equations. However, it is highly advantageous to augment the analysis using diagrammatic representations of probability distributions, called *probabilistic graphical models*.


In a probabilistic graphical model, each *node* represents a random variable (or group of random variables), and the *links* express probabilistic relationships between these variables. The graph then captures the way in which the joint distribution over all of the random variables can be decomposed into a product of factors each depending only on a subset of the variables. We shall discuss three types of graphical models:

* *Bayesian networks*, also known as *directed graphical models*, are useful for expressing causal relationships between random variables.
* *Markov random fields*, also known as *undirected graphical models*, are better suited to expressing soft constraints between random variables.
* *Fractor graph*, converted by both directed and undirected graphs, is for the purposes of solving inference problems.


## Motivation 

In order to motivate the use of directed graphs to describe probability distributions, consider first an arbitary joint distribution $$p(a, b, c)$$. By application of the product rule of probability, we can write the joint distribution in the form

$$ p(a, b, c) = p(c|a, b) p(b|a) p(a) $$

The right-hand side is represented in terms of a simple graphical model as follows. For each conditional distribution we add directed links (arrows) to the graph from the nodes corresponding to the variables on which the distribution is conditioned.

<figure>
    <img src="/assets/images/2009/directed-graphical-model.png" width="120px" />
    <figcaption>Fig. A directed graphical model representing the joint probability distribution over three variables.</figcaption>
</figure>


Let us extend the example by considering the joint distribution over $$K$$ variables given by $$p(x_1, \dots, x_K)$$. By repeated application of the product rule of probability, this joint distribution can be written as a product of conditional distributions, one for each of the variables


$$ p(x_1, \dots, x_K) = p(x_K | x_1, \dots, x_{K-1}) \dots p(x_2 | x_1) p(x_1) $$

This graph is *fully connected* because there is a link between every pair of nodes.

It is the *absence* of links in the graph that conveys interesting information about the properties of the class of distributions that the graph represents. We can now state in general terms the relationship between a given directed graph and the corresponding distribution over the variables. For a graph with $$K$$ nodes, the joint distribution is given by

$$ p(\mathbf x) = \prod_{k=1}^K p(x_K | {pa}_k)$$

where $$ {pa}_k $$ denotes the set of parents of $$ x_k $$, and $$ \mathbf x = \{x_1, \dots, x_K\} $$.


## Conditional Independence

Consider three variables $$a$$, $$b$$ and $$c$$, and suppose that the conditinal distribution of $$a$$, given $$b$$ and $$c$$, is such that it does not depend on the value of $$b$$, so that 

$$ p(a|b,c) = p(a|c) $$

We say that *$$a$$ is conditional independent of $$b$$ given $$c$$*. If we consider the joint distribution of $$a$$ and $$b$$ considered on $$c$$, it can be written if the form

$$ p(a,b|c) = p(a|b,c) p(b|c) = p(a|c) p(b|c) $$

Thus we see that, conditioned on c, the joint distribution of $$a$$ and $$b$$ factorizes into the product of the marginal distribution of a and the marginal distribution of b (again both conditioned on c). This says that the variables a and b are statistically independent, given c. 


<figure>
    <img src="/assets/images/2009/conditional-independence.png" width="120px" />
    <figure>Fig. A directed graphical model with conditional independence variables. </figure>
</figure>


Graphical models specify **a factorization of the joint distribution** over a set of variables into a product of local conditional distributions. They also define **a set of conditional independence properties** that must be satisfied by any distribution that factorizes according to the graph.

## Markov Random Fields 

### Conditional independence properties

#### Why do we need undirected graphical models?

In the case of directed graphs, it is possible to test whether a particular conditional independence property holds by applying a graphical test called d-separation. This involved testing whether or not the paths connecting two sets of nodes were ‘blocked’. The definition of blocked, however, was somewhat subtle due to the presence of paths having head-to-head nodes.

<figure>
    <img src="/assets/images/2009/head-to-head-nodes.png" width="560px">
    <figcaption>Source: http://user.it.uu.se/~thosc112/ML/le8.pdf</figcaption>
</figure>

*Undirected graphical models* define an alternative graphical semantics for probability distributions in which conditional independence is determined by simple graph separation. By removing the directionality from the links of the graph, the asymmetry between parent and child nodes is removed, and so the subtleties associated with head-to-head nodes no longer arise.

#### How to do conditional independence test?

Suppose that in an undirected graph we identify three sets of nodes, denoted $$A$$, $$B$$ and $$C$$, and that we consider the conditional independence property

$$ A ⊥ B | C $$

Consider all possible paths that connect nodes in set A to nodes in set B:

* If all such paths pass through one or more nodes in set C, then all such paths are ‘blocked’ and so the conditional independence property holds. 
* However, if there is at least one such path that is not blocked, then the property does not necessarily hold, or more precisely there will exist at least some distributions corresponding to the graph that do not satisfy this conditional independence relation. 

<figure><img src="/assets/images/2009/conditional-independence-undirected.png" width="560px" /></figure>

An alternative way to view the conditional independence test is to imagine removing all nodes in set C from the graph together with any links that connect to those nodes. We then ask if there exists a path that connects any node in A to any node in B. If there are no such paths, then the conditional independence property must hold.

### Factorization properties

A factorization rule for undirected graphs involve expressing the joint distribution $$ p(\mathbf x) $$ as a product of functions defined over sets of variables that are **local to the graph** .

#### What is the appropriate notion of locality in the undirected graphs?

Consider two nodes $$x_i$$ and $$x_j$$ that are not connected by a link, then they must be conditional independent given all other nodes in the graph. This conditinal independence property can be expressed as

$$ p(x_i, x_j|\mathbf x_{\setminus\{i,j\}}) = p(x_i|\mathbf x_{\setminus\{i,j\}}) p(x_j|\mathbf x_{\setminus\{i,j\}})$$

The factorization of the joint distribution must therefore be such that $$x_i$$ and $$x_j$$ do not appear in the same factor in order for the conditional independence property to hold for all possible distributions belonging to the graph.

This leads us to consider a graphical concept called a *clique*, which is defined as a subset of the nodes in a graph such that there exists a link between all pairs of nodes in the subset. Futhermore, a *maximal clique* is a clique such that it is not possible to include any other nodes from the graph in the set without it ceasing to be a clique. 

#### How to decompose the joint distribution?

We can therefore define the factors in the decomposition of the joint distribution to be functions of the variables in the cliques. If $$\{x_1, x_2, x_3\}$$ is a maximal clique and we define an arbitrary function over this clique, then including another factor defined over a subset of these variables would be redundant.

> Note: why there is no factor invovling variables in different cliques? 

Let us denote a clique by $$C$$ and the set of variables in that clique by $$\mathbf x_C$$ . Then the joint distribution is written as a product of *potential functions* $$\psi_C(\mathbf x_C)$$ over the maximal cliques of the graph 

$$ p(\mathbf x) = \frac 1 Z \prod_C \psi_C(\mathbf x_C) , \quad \psi_C(\mathbf x_C) \geq 0$$

Here the quantity $$Z$$, sometimes called the *partition function*, is a normalization constant and is given by

$$ Z = \sum_{\mathbf x} \prod_C \psi_C(\mathbf x_C) $$

which ensures that the distribution $$p(\mathbf x)$$ is correctly normalized. Note that we do not restrict the choice of potential functions to those that have a specific probabilistic interpretation as marginal or conditional distributions, this is contrast to the directed graphs. One consequence of the generality of the potential function $$\psi_C(\mathbf x_C)$$ is that their product will in general not be correctly normalized.

The presence of this normalization constant is one of the major limitations of undirected graphs. If we have a model with $$M$$ discrete nodes each having $$K$$ states, then the valuation of the normalization term involves summing over $$K^M$$ states and so (in the worst case) is exponential in the size of the model. The partition function is needed for parameter learning because it will be a function of any parameters that govern the potential functions $$\psi_C(\mathbf x_C)$$. However, for evaluation of local conditional distributions, the partition function is not needed because a conditional is the ratio of two marginals, and the partition function cancels between numerator and denominator when evaluating this ratio.

#### How to make a relationship between factorization and conditional independence?

> Note: 这部分不很理解。

Given the restriction that potential functions $$\psi_C(\mathbf x_C)$$ are strictly positive, we can make a precise relationship between factorization and conditional independence.

To do this we again return to the concept of a graphical model as a filter, correspoding to Figure 8.25. Consider the set of all possible distributions defined over a fixed set of variables corresponding to the nodes of a particular undirected graph. We can define $$UI$$ to be the set of such distributions that are consistent with **the set of conditional independence statements** that can be read from the graph using graph separation. Similarly, we can define $$UF$$ to be **the set of such distributions that can be expressed as a factorization of the form (8.39) with respect to the maximal cliques of the graph** . The *Hammersley-Clifford* theorem (Clifford, 1990) states that the sets UI and UF are identical.

> Note: 这段和上文 relationship between factorization and conditional independence 的关系是什么？

Because we are restricted to potential functions which are strictly positive it is convenient to express them as exponentials, so that

$$\psi_C(\mathbf x_C) = \exp \{−E (\mathbf x_C )\} $$

where $$E(\mathbf x_C)$$ is called an *energy* function, and the exponential representation is called the *Boltzmann distribution* . The joint distribution is defined as the product of potentials, and so the total energy is obtained by adding the energies of each of the maximal cliques.

In contrast to the factors in the joint distribution for a directed graph, the potentials in an undirected graph do not have a specific probabilistic interpretation. Although this gives greater flexibility in choosing the potential functions, because there is no normalization constraint, it does raise the question of **how to motivate a choice of potential function** for a particular application. This can be done by **viewing the potential function as expressing which configurations of the local variables are preferred to others** . Global configurations that have a relatively high probability are those that find a good balance in satisfying the (possibly conflicting) influences of the clique potentials. We turn now to a specific example to illustrate the use of undirected graphs.

### Illustration: Image de-noising

We can illustrate the application of undirected graphs using an example of noise removal from a binary image (Besag, 1974; Geman and Geman, 1984; Besag, 1986). Let **the observed noisy image** be described by an array of binary pixel values $$y_i \in \{−1, +1\}$$, where the index $$i = 1, \dots , D$$ runs over all pixels. We shall suppose that the image is obtained by taking **an unknown noise-free image**, described by binary pixel values $$x_i \in \{−1, +1\}$$ and randomly flipping the sign of pixels with some small probability. An example binary image, together with a noise corrupted image obtained by flipping the sign of the pixels with probability 10%, is shown in Figure 8.30. Given the noisy image, our goal is to recover the original noise-free image.
Because the noise level is small, we know that there will be **a strong correlation between $$x_i$$ and $$y_i$$**. We also know that neighbouring pixels $$x_i$$ and $$x_j$$ in an image are strongly correlated. This prior knowledge **can be captured using the Markov random field** model whose undirected graph is shown in Figure 8.31. This graph has two types of cliques, each of which contains two variables. The cliques of the form $${x_i,y_i}$$ have an associated energy function that expresses the correlation between these variables. We choose a very simple energy function for these **cliques of the form $$−\eta x_i y_i$$** where $$\eta$$ is a positive constant. This has the desired effect of **giving a lower energy (thus encouraging a higher probability) when $$x_i$$ and $$y_i$$ have the same sign and a higher energy when they have the opposite sign**.

<figure>
    <img src="/assets/images/2009/image-denoising.png" width="640px" />
</figure>
<figure>
    <img src="/assets/images/2009/image-denoising-graph-model.png" width="640px" />
</figure>

The remaining cliques comprise pairs of variables $${x_i,x_j}$$ where i and j are indices of neighbouring pixels. Again, we want the energy to be lower when the pixels have the same sign than when they have the opposite sign, and so we choose an energy given by $$−\beta x_i x_j$$ where $$\beta$$ is a positive constant.

Because a potential function is an arbitrary, nonnegative function over a maximal clique, we can multiply it by any nonnegative functions of subsets of the clique, or equivalently we can add the corresponding energies. In this example, this allows us to add an extra term $$h x_i$$ for each pixel $$i$$ in the noise-free image. Such a term has the effect of **biasing the model towards pixel values that have one particular sign in preference to the other** .

The complete energy function for the model then takes the form 

$$$(\mathbf x, \mathbf y) = h \sum_i {x_i} - \beta \sum_{\{i,j\}} x_i x_j - \eta \sum_i x_i y_i$$

which defines a joint distribution over $$\mathbf x$$ and $$\mathbf y$$ given by

$$p(\mathbf x, \mathbf y) = \frac 1 Z \exp \{-E(\mathbf x, \mathbf y)\}$$
