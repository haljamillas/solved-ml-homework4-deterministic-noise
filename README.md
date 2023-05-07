Download Link: https://assignmentchef.com/product/solved-ml-homework4-deterministic-noise
<br>



<ol>

 <li>(Lecture 13) Consider the target function <em>f</em>(<em>x</em>) = <em>e<sup>x</sup></em>. When <em>x </em>is uniformly sampled from [0<em>,</em>2], and we use all linear hypotheses <em>h</em>(<em>x</em>) = <em>w </em> <em>x </em>to approximate the target function with respect to the squared error, what is the magnitude of deterministic noise for each <em>x</em>? Choose the correct answer; explain your answer.

  <ul>

   <li>|<em>e<sup>x</sup></em>|</li>

  </ul></li>

</ol>

(<em>Hint: If you want to take page 17 of Lecture 13 for inspiration, please note that the answer on page 17 is </em><em>not exact. Here, however, we are asking you for an exact answer.</em>)

<strong>Learning Curve</strong>

<ol start="2">

 <li>(Lecture 13) Learning curves are important for us to understand the behavior of learning algorithms. The learning curves that we have plotted in lecture 13 come from polynomial regression with squared error, and we see that the expected <em>E</em><sub>in </sub>curve is always below the expected <em>E</em><sub>out </sub> Next, we think about whether this behavior is also true in general. Consider the 0<em>/</em>1 error, an arbitrary non-empty hypothesis set H, and a learning algorithm A that returns one <em>h </em>∈ H with the minimum <em>E</em><sub>in </sub>on any non-empty data set D. That is,</li>

</ol>

A(D) = argmin<em>E</em><sub>in</sub>(<em>h</em>)<em>.</em>

<em>h</em>∈H

Assume that each example in D is generated i.i.d. from a distribution P, and define <em>E</em><sub>out</sub>(<em>h</em>) with respect to the distribution. How many of the following statements are <em>always false</em>?

<ul>

 <li>ED[<em>E</em><sub>in</sub>(A(D))] <em>&lt; </em>ED[<em>E</em><sub>out</sub>(A(D))]</li>

 <li>ED[<em>E</em><sub>in</sub>(A(D))] = ED[<em>E</em><sub>out</sub>(A(D))]</li>

 <li>ED[<em>E</em><sub>in</sub>(A(D))] <em>&gt; </em>ED[<em>E</em><sub>out</sub>(A(D))]</li>

</ul>

Choose the correct answer; explain your answer.

<ul>

 <li>0</li>

 <li>1</li>

 <li>2</li>

</ul>

<h1>[d] 3</h1>

<strong>[e] </strong>1126 (seriously?)

(<em>Hint: Think about the optimal hypothesis h</em><sup>∗ </sup>= argmin<em><sub>h</sub></em><sub>∈H </sub><em>E</em><sub>out</sub>(<em>h</em>).)

<strong>Noisy Virtual Examples</strong>

<ol start="3">

 <li>(Lecture 13) On page 20 of Lecture 13, we discussed about adding “virtual examples” (hints) to help combat overfitting. One way of generating virtual examples is to add a small noise to the input vector <strong>x </strong><sup>∈ </sup>R<em><sup>d</sup></em><sup>+1 </sup>(including the 0-th component <em>x</em><sub>0</sub>) For each (<strong>x</strong><sub>1</sub><em>,y</em><sub>1</sub>)<em>,</em>(<strong>x</strong><sub>2</sub><em>,y</em><sub>2</sub>)<em>,…,</em>(<strong>x</strong><em><sub>N</sub>,y<sub>N</sub></em>) in our training data set, assume that we generate virtual examples (<strong>x</strong>˜<sub>1</sub><em>,y</em><sub>1</sub>)<em>,</em>(<strong>x</strong>˜<sub>2</sub><em>,y</em><sub>2</sub>)<em>,…,</em>(<strong>x</strong>˜<em><sub>N</sub>,y<sub>N</sub></em>) where <strong>x</strong>˜<em><sub>n </sub></em>is simply <strong>x</strong><em><sub>n </sub></em>+ and the noise vector <em> </em>∈ R<em><sup>d</sup></em><sup>+1 </sup>is generated i.i.d. from a multivariate normal distribution N(<strong>0</strong><em>,σ</em><sup>2 </sup> I<em><sub>d</sub></em><sub>+1</sub>). Here <strong>0 </strong>∈ R<em><sup>d</sup></em><sup>+1 </sup>denotes the all-zero vector and I<em><sub>d</sub></em><sub>+1 </sub>is an identity matrix of size <em>d </em>+ 1.</li>

</ol>

Recall that when training the linear regression model, we need to calculate X<em><sup>T</sup></em>X first. Define the hinted input matrix

X<em>.</em>

What is the expected value), where the expectation is taken over the (Gaussian)-noise generating process above? Choose the correct answer; explain your answer.

<ul>

 <li>X<em>T</em>X + <em>σ</em>2I<em>d</em>+1</li>

 <li>X<em>T</em>X + 2<em>σ</em>2I<em>d</em>+1</li>

 <li>2X<em>T</em>X + <em>σ</em>2I<em>d</em>+1</li>

 <li>2X<em><sup>T</sup></em>X + <em>Nσ</em><sup>2</sup>I<em><sub>d</sub></em><sub>+1</sub></li>

 <li>2X<em><sup>T</sup></em>X + 2<em>Nσ</em><sup>2</sup>I<em><sub>d</sub></em><sub>+1</sub></li>

</ul>

(<em>Note: The choices here “hint” you that the expected value is related to the matrix being inverted for regularized linear regression—see page 10 of Lecture 14. That is, data hinting “by noise” is closely related to regularization. If </em><strong>x </strong><em>contains the pixels of an image, the virtual example is a Gaussian-noise-contaminated image with the same label, e.g. </em>https://en.wikipedia.org/wiki/ Gaussian_noise. <em>Adding such noise is a very common technique to generate virtual examples for images.</em>)

<ol start="4">

 <li>(Lecture 13) Following the previous problem, when training the linear regression model, we also need to calculate X<em><sup>T</sup></em><strong>y</strong>. Define the hinted label vector . What is the expected value</li>

</ol>

E(X<em><sup>T</sup><sub>h</sub></em><strong>y</strong><em><sub>h</sub></em>), where the expectation is taken over the (Gaussian)-noise generating process above? Choose the correct answer; explain your answer.

<ul>

 <li>2<em>N</em>X<em><sup>T</sup></em><strong>y</strong></li>

 <li><em>N</em>X<em><sup>T</sup></em><strong>y</strong></li>

 <li><strong>0</strong></li>

 <li>X<em><sup>T</sup></em><strong>y</strong></li>

</ul>

<h1>[e] 2X<em><sup>T</sup></em>y</h1>

<strong>Regularization</strong>

<ol start="5">

 <li>(Lecture 14) Consider the matrix of input vectors as X (as defined in Lecture 9), and assume X<em><sup>T</sup></em>X to be invertible. That is, X<em><sup>T</sup></em>X must be symmetric positive definite and can be decomposed to QΓQ<em><sup>T</sup></em>, where Q is an orthogonal matrix (Q<em><sup>T</sup></em>Q = QQ<em><sup>T </sup></em>= I<em><sub>d</sub></em><sub>+1</sub>) and Γ is a diagonal matrix that contains the eigenvalues <em>γ</em><sub>0</sub>, <em>γ</em><sub>1</sub>, <em>…</em>, <em>γ<sub>d </sub></em>of X<em><sup>T</sup></em> Note that the eigenvalues must be positive.</li>

</ol>

Now, consider a feature transform <strong>Φ</strong>(<strong>x</strong>) = Q<em><sup>T</sup></em><strong>x</strong>. The feature transform “rotates” the original <strong>x</strong>. After transforming each <strong>x</strong><em><sub>n </sub></em>to <strong>z</strong><em><sub>n </sub></em>= <strong>Φ</strong>(<strong>x</strong><em><sub>n</sub></em>), denote the new matrix of transformed input vectors as Z. That is, Z = XQ. Then, apply regularized linear regression in the Z-space (see Lecture 12). That is, solve

1 min kZ<strong>w </strong>− <strong>y </strong><em>. </em><strong>w</strong>∈R<em>d</em>+1 <em>N</em>

Denote the optimal solution when <em>λ </em>= 0 as <strong>v </strong>(i.e. <strong>w</strong><sub>lin</sub>), and the optimal solution when <em>λ &gt; </em>0 as <strong>u </strong>(i.e., <strong>w</strong><sub>reg</sub>). What is the ratio <em>u<sub>i</sub>/v<sub>i</sub></em>? Choose the correct answer; explain your answer.

(<em>Note: All the choices are of value &lt; </em>1 <em>if λ &gt; </em>0<em>. This is the behavior of weight “decay”—</em><strong>w</strong><em><sub>reg </sub>is shorter than </em><strong>w</strong><em><sub>lin</sub>. That is why the L2-regularizer is also called the weight-decay regularizer</em>)

<ol start="6">

 <li>(Lecture 14) Consider a one-dimensional data set where each <em>x<sub>n </sub></em>∈ R and <em>y<sub>n </sub></em>∈ R.</li>

</ol>

Then, solve the following one-variable regularized linear regression problem:

<em>.</em>

If the optimal solution to the problem above is <em>w</em><sup>∗</sup>, it can be shown that <em>w</em><sup>∗ </sup>is also the optimal solution of

subject to <em>w</em><sup>2 </sup>≤ <em>C</em>

with <em>C </em>= (<em>w</em><sup>∗</sup>)<sup>2</sup>. This allows us to express the relationship between <em>C </em>in the constrained optimization problem and <em>λ </em>in the augmented optimization problem for any <em>λ &gt; </em>0. What is the relationship? Choose the correct answer; explain your answer.

(<em>Note: All the choices hint you that a smaller λ corresponds to a bigger C.</em>)

<ol start="7">

 <li>(Lecture 14) Additive smoothing (https://en.wikipedia.org/wiki/Additive_smoothing) is a simple yet useful technique in estimating discrete probabilities. Consider the technique for estimating the head probability of a coin. Let <em>y</em><sub>1</sub><em>,y</em><sub>2</sub><em>,…,y<sub>N </sub></em>denotes the flip results from a coin, with <em>y<sub>n </sub></em>= 1 meaning a head and <em>y<sub>n </sub></em>= 0 meaning a tail. Additive smoothing adds 2<em>K </em>“virtual flips”, with <em>K </em>of them being head and the other <em>K </em>being tail. Then, the head probability is estimated by</li>

</ol>

The estimate can be viewed as the optimal solution of

<em>,</em>

where Ω(<em>y</em>) is a “regularizer” to this estimation problem. What is Ω(<em>y</em>)? Choose the correct answer; explain your answer.

<ul>

 <li>(<em>y </em>+ 1)<sup>2</sup></li>

 <li>(<em>y </em>+ 0<em>.</em>5)<sup>2</sup></li>

 <li><em>y</em><sup>2</sup></li>

 <li>(<em>y </em>− 0<em>.</em>5)<sup>2</sup></li>

 <li>(<em>y </em>− 1)<sup>2</sup></li>

</ul>

<ol start="8">

 <li>(Lecture 14) On page 12 of Lecture 14, we mentioned that the ranges of features may affect regularization. One common technique to align the ranges of features is to consider a “scaling” transformation. Define <strong>Φ</strong>(<strong>x</strong>) = Γ<sup>−1</sup><strong>x</strong>, where Γ is a diagonal matrix with positive diagonal values <em>γ</em><sub>0</sub><em>,γ</em><sub>1</sub><em>,…,γ<sub>d</sub></em>. Then, conducting L2-regularized linear regression in the Z-space.</li>

</ol>

X <strong>w</strong>˜∈R<em>d</em>+1 <em>N           N</em>

is equivalent to regularized linear regression in the X-space

<em>N </em>min            <strong>w</strong>

with a different regularizer Ω(<strong>w</strong>). What is Ω(<strong>w</strong>)? Choose the correct answer; explain your answer.

<h1>[a] w<em><sup>T</sup></em>Γw</h1>

<ul>

 <li><strong>w</strong><em><sup>T</sup></em>Γ<sup>2</sup><strong>w</strong></li>

 <li><strong>w</strong><em><sup>T</sup></em><strong>w</strong></li>

 <li><strong>w</strong><em><sup>T</sup></em>Γ<sup>−2</sup><strong>w</strong></li>

 <li><strong>w</strong><em><sup>T</sup></em>Γ<sup>−1</sup><strong>w</strong></li>

</ul>

<ol start="9">

 <li>(Lecture 13/14) In the previous problem, regardless of which regularizer you choose, the optimization problem is of the form</li>

</ol>

<em>N                                                                 d</em>

min                <em>           w</em><sup>2 </sup><strong>w</strong>

with positive constants <em>β<sub>i</sub></em>. We will call the problem “scaled regularization.”

Now, consider linear regression with virtual examples.             That is, we add <em>K </em>virtual examples

(<strong>x</strong>˜<sub>1</sub><em>,y</em>˜<sub>1</sub>)<em>,</em>(<strong>x</strong>˜<sub>2</sub><em>,y</em>˜<sub>2</sub>)<em>…</em>(<strong>x</strong>˜<em><sub>K</sub>,y</em>˜<em><sub>K</sub></em>) to the training data set, and solve

! min <em>  .</em>

<strong>w</strong>

We will show that using some “special” virtual examples, which were claimed to be a possible way to combat overfitting in Lecture 13, is related to regularization, another possible way to combat overfitting discussed in Lecture 14.

Let X = [<sup>˜ </sup><strong>x</strong>˜<sub>1</sub><strong>x</strong>˜<sub>2 </sub><em>…</em><strong>x</strong>˜<em><sub>K</sub></em>]<em><sup>T</sup></em>, <strong>y</strong>˜ = [<em>y</em>˜<sub>1</sub><em>,y</em>˜<sub>2 </sub><em>…y</em>˜<em><sub>K</sub></em>]<em><sup>T</sup></em>, and B be a diagonal matrix that contains <em>β</em><sub>0</sub><em>,β</em><sub>1</sub><em>,β</em><sub>2</sub><em>,…,β<sub>d </sub></em>in its diagonals. Set <em>K </em>= <em>d</em>+1, for what X and˜ <strong>y</strong>˜ will the optimal solution of this linear regression be the same as the optimal solution of the scaled regularization problem above? Choose the correct answer; explain your answer.

<ul>

 <li>X =<sup>˜ </sup><em>λ</em>I<em><sub>K</sub>,</em><strong>y</strong>˜ = <strong>0</strong></li>

</ul>

√      √

<ul>

 <li>X =˜ <em>λ </em>            B<em>,</em><strong>y</strong>˜ = <strong>0</strong></li>

</ul>

√

<ul>

 <li>X =˜ <em>λ </em> B<em>,</em><strong>y</strong>˜ = <strong>0</strong></li>

</ul>

√

<ul>

 <li>X = B˜ <em>,</em><strong>y</strong>˜ =        <em>λ</em><strong>1</strong></li>

 <li>X = B˜ <em>,</em><strong>y</strong>˜ = <em>λ</em><strong>1</strong></li>

</ul>

(<em>Note: Both Problem 3 and this problem show that data hinting is closely related to regularization.</em>)

<strong>Leave-one-out</strong>

<ol start="10">

 <li>(Lecture 15) Consider a binary classification algorithm A<sub>majority</sub>, which returns a constant classifier that always predicts the majority class (i.e., the class with more instances in the data set that it sees). As you can imagine, the returned classifier is the best-<em>E</em><sub>in </sub>one among all constant classifiers. For a binary classification data set with <em>N </em>positive examples and <em>N </em>negative examples, what is <em>E</em><sub>loocv</sub>(A<sub>majority</sub>)? Choose the correct answer; explain your answer.</li>

</ol>

<h1>[a] 0</h1>

<ul>

 <li>1<em>/N</em></li>

 <li>1<em>/</em>2</li>

 <li>(<em>N </em>− 1)<em>/N</em></li>

 <li>1</li>

</ul>

<ol start="11">

 <li>(Lecture 15) Consider the decision stump model and the data generation process mentioned in Problem 16 of Homework 2, and use the generation process to generate a data set of <em>N </em>examples (instead of 2). If the data set contains at least two positive examples and at least two negative examples, which of the following is the tightest upper bound on the leave-one-out error of the decision stump model? Choose the correct answer; explain your answer.

  <ul>

   <li>0</li>

   <li>1<em>/N</em></li>

   <li>2<em>/N</em></li>

   <li>1<em>/</em>2</li>

   <li>1</li>

  </ul></li>

 <li>(Lecture 15) You are given three data points: (<em>x</em><sub>1</sub><em>,y</em><sub>1</sub>) = (3<em>,</em>0)<em>,</em>(<em>x</em><sub>2</sub><em>,y</em><sub>2</sub>) = (<em>ρ,</em>2)<em>,</em>(<em>x</em><sub>3</sub><em>,y</em><sub>3</sub>) = (−3<em>,</em>0) with <em>ρ </em>≥ 0, and a choice between two models: constant (all hypotheses are of the form <em>h</em>(<em>x</em>) = <em>w</em><sub>0</sub>) and linear (all hypotheses are of the form <em>h</em>(<em>x</em>) = <em>w</em><sub>0 </sub>+ <em>w</em><sub>1</sub><em>x</em>). For which value of <em>ρ </em>would the two models be tied using leave-one-out cross-validation with the squared error measure? Choose the correct answer; explain your answer.</li>

 <li>(Lecture 15) Consider a probability distribution P(<strong>x</strong><em>,y</em>) that can be used to generate examples (<strong>x</strong><em>,y</em>), and suppose we generate <em>K </em>i.d. examples from the distribution as validation examples, and store them in D<sub>val</sub>. For any fixed hypothesis <em>h</em>, we can show that</li>

</ol>

<em>.</em>

val

Which of the following is ? Choose the correct answer; explain your answer.

<ul>

 <li><em>K</em></li>

 <li>1</li>

</ul>

<strong>Learning Principles</strong>

<ol start="14">

 <li>(Lecture 16) In Lecture 16, we talked about the probability to fit data perfectly when the labels are random. For instance, page 6 of Lecture 16 shows that the probability of fitting the data perfectly with decision stumps is (2<em>N</em>)<em>/</em>2<em><sup>N</sup></em>. Consider 4 vertices of a rectangle in R<sup>2 </sup>as input vectors <strong>x</strong><sub>1</sub>, <strong>x</strong><sub>2</sub>, <strong>x</strong><sub>3</sub>, <strong>x</strong><sub>4</sub>, and a 2D perceptron model that minimizes <em>E</em><sub>in</sub>(<strong>w</strong>) to the lowest possible value. One way to measure the power of the model is to consider four random labels <em>y</em><sub>1</sub>, <em>y</em><sub>2</sub>, <em>y</em><sub>3</sub>, <em>y</em><sub>4</sub>, each in ±1 and generated by i.i.d. fair coin flips, and then compute</li>

</ol>

<em>.</em>

For a perfect fitting, min<em>E</em><sub>in</sub>(<strong>w</strong>) will be 0; for a less perfect fitting (when the data is not linearly separable), min<em>E</em><sub>in</sub>(<strong>w</strong>) will be some non-zero value. The expectation above averages over all 16 possible combinations of <em>y</em><sub>1</sub>, <em>y</em><sub>2</sub>, <em>y</em><sub>3</sub>, <em>y</em><sub>4</sub>. What is the value of the expectation? Choose the correct answer; explain your answer.

<ul>

 <li>0<em>/</em>64</li>

 <li>1<em>/</em>64</li>

 <li>2<em>/</em>64</li>

 <li>4<em>/</em>64</li>

 <li>8<em>/</em>64</li>

</ul>

(<em>Note: It can be shown that </em>1 <em>minus twice the expected value above is the same as the so-called empirical Rademacher complexity of 2D perceptrons. Rademacher complexity, similar to the VC dimension, is another tool to measure the complexity of a hypothesis set. If a hypothesis set shatters some data points, zero E</em><sub>in </sub><em>can always be achieved and thus Rademacher complexity is </em>1<em>; if a hypothesis set cannot shatter some data points, Rademacher complexity provides a soft measure of how “perfect” the hypothesis set is.</em>)

<ol start="15">

 <li>(Lecture 16) Consider a binary classifier <em>g </em>such that</li>

</ol>

<em>.</em>

When deploying the classifier to a test distribution of <em>P</em>(<em>y </em>= +1) = <em>P</em>(<em>y </em>= −1) = 1<em>/</em>2, we get

. Now, if we deploy the classifier to another test distribution <em>P</em>(<em>y </em>= +1) = <em>p</em>

instead of 1<em>/</em>2, the <em>E</em><sub>out</sub>(<em>g</em>) under this test distribution will then change to a different value. Note that under this test distribution, a constant classifier <em>g<sub>c </sub></em>that always predicts +1 will suffer from <em>E</em><sub>out</sub>(<em>g<sub>c</sub></em>) = (1−<em>p</em>) as it errors on all the negative examples. At what <em>p</em>, if its value is between [0<em>,</em>1], will our binary classifier <em>g </em>be as good as (or as bad as) the constant classifier <em>g<sub>c </sub></em>in terms of <em>E</em><sub>out</sub>? Choose the correct answer; explain your answer.

<strong>Experiments with Regularized Logistic Regression</strong>

Consider L2-regularized logistic regression with second-order polynomial transformation.

<strong>w</strong><em><sub>λ </sub></em>= argmin<em>,</em>

<strong>w</strong>

Here <strong>Φ</strong><sub>2 </sub>is the second-order polynomial transformation introduced in page 2 of Lecture 12 (with <em>Q </em>= 2), defined as

Given that <em>d </em>= 6 in the following data sets, your Φ<sub>2</sub>(<strong>x</strong>) should be of 28 dimensions (including the constant dimension).

Next, we will take the following file as our training data set D:

http://www.csie.ntu.edu.tw/~htlin/course/ml20fall/hw4/hw4_train.dat

and the following file as our test data set for evaluating <em>E</em><sub>out</sub>:

http://www.csie.ntu.edu.tw/~htlin/course/ml20fall/hw4/hw4_test.dat

We call the algorithm for solving the problem above as A<em><sub>λ</sub></em>. The problem guides you to use LIBLINEAR (https://www.csie.ntu.edu.tw/~cjlin/liblinear/), a machine learning packaged developed in our university, to solve this problem. In addition to using the default options, what you need to do when running LIBLINEAR are

<ul>

 <li>set option -s 0, which corresponds to solving regularized logistic regression</li>

 <li>set option -c C, with a parameter value of C calculated from the <em>λ </em>that you want to use; read README of the software package to figure out how C and your <em>λ </em>should relate to each other</li>

 <li>set option -e 0.000001, which corresponds to getting a solution that is really really close to the optimal solution</li>

</ul>

LIBLINEAR can be called from the command line or from major programming languages like python. If you run LIBLINEAR in the command line, please include screenshots of the commands/results; if you run LIBLINEAR from any programming language, please include screenshots of your code.

We will consider the data set as a <em>binary classification problem </em>and take the “regression for classification” approach with regularized logistic regression (see Page 6 of Lecture 10). So please evaluate all errors below with the 0/1 error.

<ol start="16">

 <li>Select the best <em>λ</em><sup>∗ </sup><em>in a cheating manner </em>as</li>

</ol>

argmin             <em>E</em><sub>out</sub>(<strong>w</strong><em><sub>λ</sub></em>)<em>.</em>

log<sub>10 </sub><em>λ</em>∈{−4<em>,</em>−2<em>,</em>0<em>,</em>2<em>,</em>4}

Break the tie, if any, by selecting the largest <em>λ</em>. What is log<sub>10</sub>(<em>λ</em><sup>∗</sup>)? Choose the closest answer; provide your command/code.

<ul>

 <li>−4</li>

 <li>−2</li>

 <li>0</li>

 <li>2</li>

 <li>4</li>

</ul>

<ol start="17">

 <li>Select the best <em>λ</em><sup>∗ </sup>as</li>

</ol>

argmin             <em>E</em><sub>in</sub>(<strong>w</strong><em><sub>λ</sub></em>)<em>.</em>

log<sub>10 </sub><em>λ</em>∈{−4<em>,</em>−2<em>,</em>0<em>,</em>2<em>,</em>4}

Break the tie, if any, by selecting the largest <em>λ</em>. What is log<sub>10</sub>(<em>λ</em><sup>∗</sup>)? Choose the closest answer; provide your command/code.

<ul>

 <li>−4</li>

 <li>−2</li>

 <li>0</li>

 <li>2</li>

 <li>4</li>

</ul>

<ol start="18">

 <li>Now split the given training examples in D to two sets: the first 120 examples as D<sub>train </sub>and 80 as D<sub>val</sub>. (<em>Ideally, you should randomly do the </em>120<em>/</em>80 <em> Because the given examples are already randomly permuted, however, we would use a fixed split for the purpose of this problem</em>). Run A<em><sub>λ </sub></em>on <em>only </em>D<sub>train </sub>to get <strong>w</strong><em><sub>λ</sub></em><sup>− </sup>(the weight vector within the <em>g</em><sup>− </sup>returned), and validate <strong>w</strong><em><sub>λ</sub></em><sup>− </sup>with D<sub>val </sub>to get). Select the best <em>λ</em><sup>∗ </sup>as</li>

</ol>

<em>.</em>

Break the tie, if any, by selecting the largest <em>λ</em>. Then, estimate <em>E</em><sub>out</sub>(<strong>w</strong><em><sub>λ</sub></em><sup>−</sup><sub>∗</sub>) with the test set. What is the value of <em>E</em><sub>out</sub>(<strong>w</strong><em><sub>λ</sub></em><sup>−</sup><sub>∗</sub>)? Choose the closest answer; provide your command/code.

<ul>

 <li>0<em>.</em>10</li>

 <li>0<em>.</em>11</li>

 <li>0<em>.</em>12</li>

 <li>0<em>.</em>13</li>

 <li>0<em>.</em>14</li>

</ul>

<ol start="19">

 <li>For the <em>λ</em><sub>∗ </sub>selected in the previous problem, compute <strong>w</strong><em><sub>λ</sub></em>∗ by running A<em><sub>λ</sub></em>∗ with the full training set D. Then, estimate <em>E</em><sub>out</sub>(<strong>w</strong><em><sub>λ</sub></em>∗) with the test set. What is the value of <em>E</em><sub>out</sub>(<strong>w</strong><em><sub>λ</sub></em>∗)? Choose the closest answer; provide your command/code.

  <ul>

   <li>0<em>.</em>10</li>

   <li>0<em>.</em>11</li>

   <li>0<em>.</em>12</li>

   <li>0<em>.</em>13</li>

   <li>0<em>.</em>14</li>

  </ul></li>

 <li>Now split the given training examples in D to five folds, the first 40 being fold 1, the next 40 being fold 2, and so on. Again, we take a fixed split because the given examples are already randomly permuted. Select the best <em>λ</em><sup>∗ </sup>as</li>

</ol>

argmin             <em>E</em><sub>cv</sub>(A<em><sub>λ</sub></em>)<em>.</em>

log<sub>10 </sub><em>λ</em>∈{−4<em>,</em>−2<em>,</em>0<em>,</em>2<em>,</em>4}

Break the tie, if any, by selecting the largest <em>λ</em>. What is the value of <em>E</em><sub>cv</sub>(A<em><sub>λ</sub></em>∗) Choose the closest answer; provide your command/code.

<ul>

 <li>0<em>.</em>10</li>

 <li>0<em>.</em>11</li>

 <li>0<em>.</em>12</li>

 <li>0<em>.</em>13</li>

 <li>0<em>.</em>14</li>

</ul>