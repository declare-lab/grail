from utils.reward_utils import check_correctness, is_conversational


def outcome_reward(completions, **kwargs):
    if is_conversational(completions[0]):
        completion_texts = [completion[0]["content"] for completion in completions]
    else:
        completion_texts = completions

    rewards = []
    gt_answers = [answer.strip() for answer in kwargs["solution"]]

    for completion, gt in zip(completion_texts, gt_answers):
        reward = check_correctness(completion, gt)
        rewards.append(reward)

    return rewards
