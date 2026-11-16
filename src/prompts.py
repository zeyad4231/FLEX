"""LM prompts, copied from the paper's Appendix 7.1 (pages 15-21).

Only the {placeholders} are filled in by flex.py. Changes from the paper are
marked "ADAPTED" and listed in DEVIATIONS.md.
"""

# ------------------------------------------------------------- system prompt
SYSTEM_2D = """An agent is in a 2D gridworld environment of size 9x9. (0,0) is the top left corner and (8, 8) is the bottom right corner (note this is not a typical coordinate grid).
The environment config has the following attributes:
    X (int): width of the environment
    Y (int): height of the environment
    starts (List[Tuple[int]]): starting coordinate of the agent
    goals pos (List[Tuple[int]]): coordinate position of the goal
    goals color (str): color of the goal
    goals type (str): object type of the goal
    objects{1-3} pos (List[Tuple[int]]): position(s) of other objects in the environment. This will be empty if there is no object in that slot.
    objects{1-3} color (str): color of the corresponding object at the position objects{1-3}. This will be empty if there is no object in that slot.
    objects{1-3} type (str): object type of the corresponding object at the position objects{1-3}. This will be empty if there is no object in that slot."""

# ------------------------------------------------------------------- ADD
ADD_GENERATE_FEATURE = """This is an environment config: {env}

The following is an instruction for a task that the agent needs to complete in this environment. instruction = {instruction}

The following is a history of corrections that the agent has been provided. history = {history}

The following is the most recent addendum to the original instruction after the agent asked for more clarification about the task. addendum = {correction}

This is the current list of features that we already know: {features}.

The feature list contains computable numeric values (e.g. "distance from agent to object X") derived from scene state. These features will be used by a downstream model that has NO access to the instruction, corrections, or addendum text -- only to the feature values.

Consider adding a new feature to this list to explain all parts of the instruction and addendum. Carefully consider all parts of the addendum, especially if multiple objects are mentioned.

1. If adding a new feature would not provide any new information that helps explain the instruction or addendum -- for example, if the existing features already capture all relevant objects, properties, or relationships -- do not propose any additional feature and return "NO NEW FEATURE". Do NOT propose a feature merely because an object exists in the environment or because it could be generally relevant in other contexts -- only if the instruction or addendum clearly implies that object or property matters for the task. If the existing features already cover all relevant entities or relationships mentioned or implied by the instruction and addendum, respond exactly with: NO NEW FEATURE

Add at most one sentence of explanation with specific references to the addendum

2. Only propose a new feature if it clearly adds missing information needed to explain the instruction or addendum. You may provide the name of the specific feature in this environment that explains the instruction and addendum. If there are multiple of the same type of object, please give the average distance to all objects of that type.

Do not mix multiple types or objects in the feature. Please specify a feature (only one feature e.g. if there are multiple possible explanations choose only one most likely object) that we can write code to compute from the position of the agent to other objects in the scene. Please give a different feature in your answer (e.g. if distance to goal feature is already in the list, do not give distance to goal again). Ground your answer in visible objects in the scene, and give an answer (only 1 specific answer (e.g. distance to heavy objects)) that we can compute relative to the position of the agent.

Follow your answer with one newline and at most one sentence of explanation with specific references to the addendum."""

# ------------------------------------------------------------- GENERALIZE
GEN_EXTRACT_INTENT = """This is an environment config: {env}

The following is an instruction for a task that the agent needs to complete in this environment. instruction = {instruction}

The following is a history of corrections that the agent has been provided. history = {history}

The following is the most recent addendum to the original instruction after the agent asked for clarification about the task. addendum = {correction}

Extract the intent behind the human's addendum in at most 2 sentences, explaining the purpose or reasoning behind the addendum. Focus on what the agent needs to achieve and why the human may have said this particular addendum, including any important context or conditions that affect the task. Do not base the explanation on specific locations, as this may change in later scenarios."""

GEN_FEATURE_NAMES = """A robot is in a 2D gridworld environment with the following objects: {env}

The following is an instruction for a task that the agent needs to complete in this environment. instruction = {instruction}

The following is a history of corrections that the agent has been provided. history = {history}

The following is an addendum to the original instruction after the agent asked for clarification about the task. addendum = {correction}

{intent}

The robot has the following set of features: {features}

For each feature, please determine whether the intent is semantically relevant for that feature.

If yes (and only if yes), generalize the feature name using properties of objects in the environment that directly reflect the semantic intent of the instruction or addendum, without creating spurious correlations.

Do not generalize based on properties that are not directly connected to the instruction intent. Do not generalize to multiple properties. Feature names may remain unchanged if they already sufficiently capture the semantic intent. Feature names may remain unchanged if they are already sufficiently general. Do not simply rename a feature that points to the same property. Features that share the same semantic relevance should be generalized to the same name. Only generalize using object properties explicitly listed in the environment.

Return a newline-delimited list of the original feature names, followed by "->", followed by either NO CHANGE or a generalized feature name. For example
feature a -> feature letter
feature b -> feature letter
feature letter -> feature letter
feature 1 -> NO CHANGE"""

# ------------------------------------------------------------------- MASK
MASK_ELEMENT = """This is an environment config: {env}

The following is an instruction for a task that the agent needs to complete in this environment. instruction = {instruction}

The following is a history of corrections that the agent has been provided. history = {history}

The following is an addendum to the original instruction after the agent asked for clarification about the task. addendum = {correction}

The current available features are: {features}

We are evaluating whether {feature}
is part of the minimal set of features needed for the agent to compute its reward given that it has NO access to the instruction, corrections, or addendum text -- only to the feature values. Please return either a 1 or 0 for the following feature: feature = {feature}, where 1 means that the agent uses this feature to calculate its reward and 0 means that the agent does not need this feature when calculating its reward.

Please reason step by step. In particular, consider whether the feature provides precise and relevant information for the task. Also consider whether the feature might lead the agent to treat multiple objects the same way, even if only one of them is relevant to the instruction.

For example, for the instruction go to a book and the features ["book dist", "table dist", "chair dist"], the minimal set would include only the features that refer specifically to the object mentioned in the instruction.

Provide your answer as ONLY a single integer. Do not include any surrounding explanation or text."""

# ---------------------------------------------------------- code generation
# ADAPTED: the paper's generic template shows the Franka signature
# (environment, trajs, robot_env); for 2D we use the 2D inputs and example
# from the same appendix (pages 19-21). The feature name is passed explicitly
# because our LLM calls are single-turn (the paper says "the feature identified above").
EXAMPLE_DISTANCE_2D = '''For example, for "agent distance to book objects" feature, you may write code

```python
from flexible_features.utils.env_utils import get_objects_with_property

def agent_distance_to_book_objects(environment, trajectory, r=3.0):
    # environment is the ground truth environment

    # returns the position(s) of the object(s) with property "book"
    book_objects_pos = get_objects_with_property("book", environment)

    # initialize the feature value
    book_objects_feat = np.zeros(1)

    # iterate through the trajectory
    for i in range(0, len(trajectory) - 1, 2):

        # iterate through the book objects
        for book_object in book_objects_pos:

            # calculate the dist between the agent and the book object (x)
            dx = abs(trajectory[i] - book_object[0])

            # calc the distance between the agent and the book object (y)
            dy = abs(trajectory[i + 1] - book_object[1])
            dist = np.linalg.norm([abs(dx), abs(dy)])

            # Cap distance at r (saturate)
            bump_dist = min(dist, r)
            book_objects_feat += bump_dist

    # return the average distance to the book object(s) for each trajectory
    feature_value = book_objects_feat / len(book_objects_pos)
    return feature_value  # array with dtype=np.float64
```'''

GENERATE_CODE_2D = """This is an environment config: {env}

The feature identified above is: {feature_name}

Based on the feature identified above, write a Python method that takes as input:
* the ground truth environment features specified in the config (dict)
* a trajectory (List[float]) representing the agent's position over time as a list of x,y coordinates (e.g. [0.0 0.0 0.0 1.0] means the agent started at (0,0) and moved to (0,1). The trajectory should be of length 34 (representing 17 timesteps where each timestep has a x,y coordinate for the agent's position)

and outputs:
* the cumulative value for the feature over the course of the entire trajectory (averaged over multiple objects if present) (array with dtype=np.float64).

In your method, you may use get_objects_with_property(property: str, environment: dict), which takes a property name and the environment dict, and returns a list of [x, y] positions for all objects in the environment that have that property, type, or name. You must import this function from flexible_features.utils.env_utils.

{example}

Only include one feature (e.g. distance to book) and not a general feature (e.g. distance to object) in your answer. Do not use objects or properties that do not exist in this environment. Please write code that is generalizable to new environments (for example, the new environment may have objects in different positions than the current environment)

Output your answer in form:
Feature name: "feature name"
Feature code:
```python
import ...
def feature_name(environment, trajectory):
    ...
    return feature_value
```

Please write code for all the features you have identified, and ensure the code is executable. Make sure your code is generalizable to new environments. Don't assume you are given any other information about the environment other than what is explicitly specified in the environment config. Do not include any surrounding explanation or text."""

CODE_RETRY = """

Your previous code failed with this error:
{error}
Please fix it and answer again in the same format."""

# ------------------------------------------------------------- baselines
# Appendix 7.2. MFE and USF use the same code-generation and masking prompts as FLEX.
MFE_EXHAUSTIVE = """This is an environment config: {env}

Use this to give specific feature(s) in that environment that explains what are relevant properties in the environment. If there are multiple of the same object, please give the average distance to all objects of that type.

Please specify feature(s) that we can write code to compute from the position of the agent to other objects in the scene. This is the current list of features that we already know: {features}. Please give different features in your answer (e.g. if distance to goal feature is already in the list, do not give distance to goal again).

Please give your confidence for each feature between 0 and 1. You may ground your answer in visible objects in the scene, and give answers that we can compute relative to the position of the agent."""

USF_UPDATE = """This is an environment config: {env}

The following is an instruction for a task that the agent needs to complete in this environment. instruction = {instruction}

The following is a history of corrections that the agent has been provided. history = {history}

The following is the most recent addendum to the original instruction after the agent asked for clarification about the task. addendum = {correction}

Based on this, create a feature list that contains computable numeric values (e.g. "distance from agent to object X") derived from scene state. These features will be used by a downstream model that has NO access to the instruction, corrections, or addendum text -- only to the feature values.

If there are multiple of the same object, please give the average distance to all objects of that type. Please specify feature(s) that we can write code to compute from the position of the agent to other objects in the scene. Every feature in your answer must be unique (e.g. if one feature is distance to goal, do not create another feature that is distance to goal again).Please give your confidence for each feature between 0 and 1. Ground your answer in visible objects in the scene, and give answers that we can compute relative to the position of the agent."""
