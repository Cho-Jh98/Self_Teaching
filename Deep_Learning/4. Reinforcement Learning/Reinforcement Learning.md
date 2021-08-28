# Reinforcement Learning



## Application

1. Artificial intelligence algorithms capable of playing games

   ex) Chess, Go, Mario

2. Artificial intelligence algorithms capable of learning basically anything

   ex) Self-driving car

3. Finance : Q-learning algorithm is able to learn an optimal trading strategy with one simple instruction

   → Maximize the value of our portfolio

4. Logistics : Vehicle routing problem



Markov-Decision Process, Deep Q learning

### Reinforcement Learning + Deep Learning → Extremely powerful



## Markov Decision Process (MDP)



In machine learning, environment is typically formulated as a **Markov Decision Process** which is extended version of a Markov Chain



1. Set of states : **S**
   * In chess, a state is given configuration of the board
2. Set of Actions : **A** (for example: up, down, left, right)
   * Every possible move in chess or go or tic-tac-toe
3. **Conditional distribution** of the next state
   * The next state depends of the actual one exclusively **"Markov-Property"**
   * Exclusively in current stage, not the previous stage
   * **P(s'|s,a)** is transition probability matrix
   * What will be the next **S'** state after the agent make action **A** while being in the state **S**
4. **R(s,s')** is the reward function of transistioning from state **S** to **S'**
5. **𝜸** gamma discount factor : Prefer rewards now not in the future

So we are going to formulate the problem as Markov-Decision Process



## Type of learning of reinforcement learning



### Finding path

* We do not have all the dataset at all!!

  * The artificial intelligence agent will interact with the environment (=states)
     and figure out what to do (=action)

  <img src="https://user-images.githubusercontent.com/84625523/126866622-c9733da7-e8a8-4511-b4ad-9262fe87596e.png" style="zoom:33%;" />

  * Upon doing actions that agent receives a reward or penalty

    ex) Finding the shortest path : there are states (Current place) and actions (where we are going to go)



### Tic-Tac-Toe

Reinforced learning can be used to learn how to play this game
→  environment : cells on the 3x3 board
→  agents (computer or human player) makes a move according to the board states
→  Actions : where to put X or O
→  Eventually the game will end, agents will receive reward(win) or penalty(loose)

* Agents initially plays poorly but after the training procedure it will be able to choose the right actions/moves



## Mathmetics in MDP



### Example_Tic-Tac-Toe

* Consider we are playing tic-tac-toe

* The next step depends on the actual step exclusively. and the formula would be as fallows 

  > **P( s(t+1) | s(t), s(t-1), s(t-2) ... s(1) ) = P( s(t+t) | s(t) )**

  This is **Markov-Property**

* The system does not have any memory.
  After algorithm reach a state, the past history should not affect the next transitions and rewards

* Set of states : these are the configurations we represent them with a tree like structure

  > <img src="https://user-images.githubusercontent.com/84625523/126866986-c106ff86-fcbc-4230-b845-7d8b23266c0c.png" style="zoom:33%;" />

* Actions : **X** or **O** on whatever cell you prefer

* Reward : **+1** if the agent wins or **-1** if it loses



#### Deterministic vs Stochastic environment

* **Deterministic environment** : There is only one state **S'** with probability **1** and all others with **0**
  * ex) Chess, tic-tac-toe, Shortest path problem
  * From state **S**, you can pick certain indexed cell **(Action)**, and of course it will be deterministic and **S'** will be the result of this action



* **Stochastic environment** : Involves randomness like a coin toss
  * ex) Blackjack as a MDP
  * Have to deal with distibutions in these cases





#### Discount factor 𝜸(gamma)

* Numerical value within the range [0, 1], which represents the relative importance between immediate and future rewards
* **The algorithm prefers reward now, not in the future**
* 𝜸 makes sure...
  * that the algorithm will converge : do not end up with infinite future rewards
  * that the algortihm will be greedy : if there is a chance to make a good step/decision, it will do it.





### Equations

* Prerequisites : There **is** a final or terminal state!!
  * ex) agent will win or lose when playing chess or have managed to find the optimal shortest path



#### Cumulative reward

 Cumulative Reward : <img src="https://user-images.githubusercontent.com/84625523/126867281-8ede8403-da5a-485f-8300-51888ae39bb2.png" style="zoom:30%;" />

* It is the discounted sum of future rewards throughout an episode(states). It means all the steps until we find the final state.

> <img src="https://user-images.githubusercontent.com/84625523/126867338-69f6c69a-792d-4e70-9c0f-29a177326b59.png" style="zoom:33%;" />

* Every state assosiated with given rewards.
  * 𝜸 = 1 : Rewards are same between both current state and future states
  * 𝜸 = 0.1 : Focusing rewards in the current state than future states (since it is exponential)



#### Policy

* Policy : 𝝅(s)
  * A policy is the agent's strategy to choose an actions at each state **S**
  * **Optimal policy** : Maximize the expectation of cumulative reward
* The **objective** of reinforcement learning is to train an agent such that the agent learns a policy as close as posible to the optimal policy.



#### Value Function

Value Function : V(s)

* This function represents  how good is a state **s** for an agent to be. It is the expected total reward for an agent starting from state **s**. 
  * Of course it depends on the **𝝅(s)** policy, by which the agent decide what action to perform.

* Formula

  > <img src="https://user-images.githubusercontent.com/84625523/126867607-0ceccd2a-f9d4-412f-8e7e-4e5407b53292.png" style="zoom:33%;" />

  * Expected value of the discounted sum of reward(**𝝅**)

  * In every single state, we need to know which action to perform, and it is determined by policy. Without policy, we don't know what to do. However, if we have policy, we can calculate expected value of sum of reward.

  * **Optimal Value Function** : It has higher value than all other value function.

    > <img src="https://user-images.githubusercontent.com/84625523/126867737-e4fb2faf-e553-4382-80e8-31ca64887970.png" style="zoom:33%;" />

  * ex) The aim is to find the shortest path from the red cell to green cell
    * Every cell has a **V(s)** value
    * Where **V(s)** is the expected total reward starting from **S**



#### Value function vs Policy

* **𝝅*(s)** : Optimal policy-function

  > <img src="https://user-images.githubusercontent.com/84625523/126867829-38f62ab7-5563-4149-b98e-dff053e355ed.png" style="zoom:33%;" />

  * So it basically means optimal policy function( **𝝅*(s) **) will generate optimal value function( **V*(s) **)



* Q-Function : **Q*(s, a)**

  * The opimal Q-function yields the expected total reward which is received by an agent starting in state **s** when picks up an action **a**

  > <img src="https://user-images.githubusercontent.com/84625523/126867901-7e9e4725-6880-4b05-bc47-ee92cb45d3d2.png" style="zoom:33%;" />

  * Value function returns a value, whereas policy function returns an action.
    * To be more specific, policy function assosiates to every single action in the environment
    * Value function defines how good a given state is, policy defines an action what to do in given state.
    * And of course optimal functions will give optimal decision, which is best action.
  * So the policy function returns an action which optimizes (maximizes) the value function. 





### Illustration with shortest path finding algorithm

#### Policy function illustration

> <img src="https://user-images.githubusercontent.com/84625523/126938817-bcd0d96b-1a63-4ac0-920c-d5f2fedaaa23.png" style="zoom:50%;" />

* **s**,state : a cell on the board (**row_index and col_index**)

* **a**, action : up, down, left, right

* **𝝅*(s)**, optimal policy : suggest what **a** to make → arrows are the suggested action accordingly

  **-1** : panelty
  **+1** : reward

* Agent will learn the optimal policy that avoids that's going to get -1, and goes to +1 state.



#### Value function illustration

> <img src="https://user-images.githubusercontent.com/84625523/126939297-81aa9711-6891-45d5-baac-37102c475e34.png" style="zoom:33%;" />

* expected total reward for an agent starting from each state **s**
* Higher value as it get close to **+1** → higer expected reward



* The optimal policy is not necessarily to fallow

  → The "gradient" so the direction of highest value in neighborhood

* Stochastic case : The movement is random
  * If it wants to go up : There is some changes that it will go left and right as well
  * 80% up + 10% left + 10% right



#### Q* function illustration

> <img src="https://user-images.githubusercontent.com/84625523/126939777-80259892-b82c-447d-be0d-1ba3fb6ce6c2.png" style="zoom:33%;" />

* There are 4 actions to make:
  * up, down, left, right
* In this case : up has 0.9, right : 0.6, down : -0.4, left : 0.7
  * down: goes to -1 penalty → gets negative value
* If we get the maximum value of **Q*(s,a)** function, we can calculate value of **V*(s)**





### Bellman-equation

* Object : maximize the reward

* **Q*(s)** yields expected total reward received by an agent starting in **s** and picks an action

  > #### Q*(s, a) = R(s, a)  +  𝜸E[V\*(s')]

  * R(s, a) : immediate reward in state **s** after action **a**
  * 𝜸E[V\*(s')] : discounted expected future reward after the transition to s' (after action)



> Since E[V\*(s')] is sum of p(s'|s, a) V\*(s') for every s' by calcuation of expactation,
>
> 𝜸E[V\*(s')] is sum of p(s'|s, a) V\*(s') times 𝜸
>
> and since V\*(s) is maximum value of Q\*(s, a), we get the equation
>
> V\*(s) = max_a[R(s, a) + sum of p(s'|s, a) V\*(s') times 𝜸]
>
> which is the Bellman-equation



* Ultimate goal is to maximize the reward function!!
  reward in current state and next state



#### How to use?

1. Value iteration
2. Policy iteration
3. Q-learning



* **value and policy iteration**

  * **assumes that agent knows the MDP model of the world**

    such as the **P(s'|s, a)**, the probability distribution and **R(s', s)**, the reward function

  * offline-learning : agent can plan its actions, given that knowledge about the environment before interacting with it

* **Q-learning**

  * model free approach : agent knows nothing about MDP model, knows the possible state and action exclusively
  * Online-learning : agent improves its behavior through learning from the history of interactions with the environment
