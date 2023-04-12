### Creating ping pong db

agent queries:

```python
def run_agent_with_memory():
    agent = initialize_agent_with_memory()

    p = """
I want to store information about my ping pong games.
I want to store who played the games, who won, and what the score was.
Figure out how to store this in the database and set up the database as such.
"""
    agent.run(input=p)

    p = """
Peyton and Lasse played ping pong. Lasse won 21-15. Record this information.
"""
    agent.run(input=p)

    p = """
Peyton and Lasse played ping pong. Peyton won 21-19. Record this information.
"""
    agent.run(input=p)

    p = """
Peyton and Lasse played ping pong. Lasse won 21-20. Record this information.
"""
    agent.run(input=p)

    p = """
Who has won more ping pong games?
"""
    agent.run(input=p)

    p = """
What's the total number of points scored by all ping pong players?
"""
    agent.run(input=p)
```

agent responses:

```
> Entering new AgentExecutor chain...
Thought: First, I need to create a schema for storing ping pong game information.
Thought: I will create a table with columns for game ID, player 1, player 2, winner, player 1 score, and player 2 score.

Action: SQL executor plugin
Action Input: 
CREATE TABLE ping_pong_games (
    game_id SERIAL PRIMARY KEY,
    player_1 VARCHAR(255) NOT NULL,
    player_2 VARCHAR(255) NOT NULL,
    winner VARCHAR(255) NOT NULL,
    player_1_score INT NOT NULL,
    player_2_score INT NOT NULL
);
Observation: Query executed successfully.
Thought:I now know the final answer.
Final Answer: I have created a table called "ping_pong_games" with the following schema:

- game_id (SERIAL PRIMARY KEY)
- player_1 (VARCHAR(255) NOT NULL)
- player_2 (VARCHAR(255) NOT NULL)
- winner (VARCHAR(255) NOT NULL)
- player_1_score (INT NOT NULL)
- player_2_score (INT NOT NULL)

You can now store your ping pong game information in this table.

> Finished chain.

> Entering new AgentExecutor chain...
Thought: I need to insert a new row into the ping_pong_games table with the given information.
Action: SQL executor plugin
Action Input: INSERT INTO ping_pong_games (player_1, player_2, winner, player_1_score, player_2_score) VALUES ('Peyton', 'Lasse', 'Lasse', 15, 21);
Observation: Query executed successfully.
Thought:I now know the final answer
Final Answer: I have recorded the ping pong game between Peyton and Lasse with Lasse winning 21-15 in the database.

> Finished chain.


> Entering new AgentExecutor chain...
Thought: I need to insert the game information into the ping_pong_games table.
Action: SQL executor plugin
Action Input: INSERT INTO ping_pong_games (player_1, player_2, winner, player_1_score, player_2_score) VALUES ('Peyton', 'Lasse', 'Peyton', 21, 19);
Observation: Query executed successfully.
Thought:I now know the final answer.
Final Answer: I have recorded the ping pong game between Peyton and Lasse with Peyton winning 21-19 in the database.

> Finished chain.


> Entering new AgentExecutor chain...
Thought: I need to insert this game record into the ping_pong_games table.
Action: SQL executor plugin
Action Input: INSERT INTO ping_pong_games (player_1, player_2, winner, player_1_score, player_2_score) VALUES ('Peyton', 'Lasse', 'Lasse', 21, 20);
Observation: Query executed successfully.
Thought:I now know the final answer.
Final Answer: I have recorded the ping pong game between Peyton and Lasse with Lasse winning 21-20 in the database.

> Finished chain.


> Entering new AgentExecutor chain...
Thought: I need to count the number of games won by each player in the ping_pong_games table.
Action: SQL executor plugin
Action Input: SELECT winner, COUNT(*) as games_won FROM ping_pong_games GROUP BY winner;
Observation: ('Lasse', 2)
('Peyton', 1)

Thought:I now know the final answer
Final Answer: Lasse has won more ping pong games.

> Finished chain.


> Entering new AgentExecutor chain...
Thought: To find the total number of points scored by all ping pong players, I should sum up the player_1_score and player_2_score columns from the ping_pong_games table.
Action: SQL executor plugin
Action Input: SELECT SUM(player_1_score) + SUM(player_2_score) as total_points FROM ping_pong_games;
Observation: (Decimal('117'),)

Thought:I now know the total number of points scored by all ping pong players.
Final Answer: The total number of points scored by all ping pong players is 117.
```