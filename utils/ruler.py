#@title Sample RULER evaluation

import art
from art.rewards import ruler_score_group

# Test RULER with a simple example
base_messages = [
    {"role": "system", "content": "You count numbers using numeric symbols."},
    {"role": "user", "content": "Count to 10."},
]

good_trajectory = art.Trajectory(
    messages_and_choices=[
        *base_messages,
        {"role": "assistant", "content": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"},
    ],
    reward=0,
)

mediocre_trajectory = art.Trajectory(
    messages_and_choices=[
        *base_messages,
        {
            "role": "assistant",
            "content": "one, two, three, four, five, six, seven, eight, nine, ten",
        },
    ],
    reward=0,
)

bad_trajectory = art.Trajectory(
    messages_and_choices=[
        *base_messages,
        {"role": "assistant", "content": "a, b, c, d, e, f, g, h, i, j"},
    ],
    reward=0,
)

sample_group = art.TrajectoryGroup(
    trajectories=[
        good_trajectory,
        mediocre_trajectory,
        bad_trajectory,
    ]
)

# if __name__ =="__main__":

#     judged_group = await ruler_score_group(sample_group, "openai/o4-mini", debug=True)
#     assert judged_group is not None

#     # Display rankings
#     sorted_trajectories = sorted(
#         judged_group.trajectories, key=lambda t: t.reward, reverse=True
#     )
#     for rank, traj in enumerate(sorted_trajectories, 1):
#         messages = traj.messages()
#         print(f"\nRank {rank}: Score {traj.reward:.3f}")
#         print(f"  Response: {messages[-1]['content'][:50]}...")