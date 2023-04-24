import keyboard

messages = [
"""
Human $ show me the current sessions on my cluster
""",
"""
> Entering new AgentExecutor chain...
Thought: I need to query the current sessions on the cluster.
Action: CockroachDB Docs QA System
Action Input: How can I view the current sessions on my CockroachDB cluster?
Observation: To view the current sessions on your CockroachDB cluster, follow these steps:

1. Access the DB Console by going to http://localhost:8080 (replace "localhost" with your DB Console address if it's not running locally).
2. Click on **SQL Activity** in the left-hand navigation of the DB Console.
3. Click the **Sessions** tab.

The Sessions page of the DB Console will display details of all open sessions in the cluster.
""",
"""
Thought:I now know the final answer
Final Answer: To view the current sessions on your CockroachDB cluster, access the DB Console by going to http://localhost:8080 (replace "localhost" with your DB Console address if it's not running locally). Click on **SQL Activity** in the left-hand navigation of the DB Console, and then click the **Sessions** tab. The Sessions page of the DB Console will display details of all open sessions in the cluster.

> Finished chain.
""",
"""
Human $ use sql to return this information to me
""",
"""
> Entering new AgentExecutor chain...
Thought: I need to run a SQL query to get the active sessions information from the cluster.
Action: CockroachDB Docs QA System
Action Input: How can I get the list of active sessions using SQL in CockroachDB?
Observation: You can get the list of active sessions in CockroachDB using the `SHOW SESSIONS` statement. Here's an example:

```sql
> SHOW SESSIONS;
```

This will list all currently active sessions across the cluster, including details about each session.
""",
"""
Thought:I now know the SQL query to get the list of active sessions.
Action: SQL executor plugin
Action Input: SHOW SESSIONS;
Observation: (1, '1758e3e8d56021080000000000000001', 'root', '127.0.0.1:62427', '', 'SHOW CLUSTER SESSIONS', '', datetime.datetime(2023, 4, 24, 14, 4, 9, 974568), datetime.datetime(2023, 4, 24, 14, 4, 9, 975291))
""",
"""
Thought:I now know the current sessions on the cluster.
Final Answer: There is currently one active session on your cluster. The session has the following details: Session ID: 1758e3e8d56021080000000000000001, User: root, Client Address: 127.0.0.1:62427, Application Name: "", SQL Statement: "SHOW CLUSTER SESSIONS", Start Time: 2023-04-24 14:04:09.974568, Transaction Start Time: 2023-04-24 14:04:09.975291
""",
"""
Human $ I play a lot of ping pong and want to store my games. Create a system that will store each player's name, their score, and who won.
""",
"""
> Entering new AgentExecutor chain...
Thought: I need to create a table to store ping pong game data.
Action: SQL executor plugin
Action Input: CREATE TABLE ping_pong_games (id SERIAL PRIMARY KEY, player1_name TEXT NOT NULL, player1_score INT NOT NULL, player2_name TEXT NOT NULL, player2_score INT NOT NULL, winner_name TEXT NOT NULL);
Observation: Query executed successfully.
""",
"""
Thought:I now know the final answer.
Final Answer: I have created a table called "ping_pong_games" with the following columns: id (unique identifier), player1_name, player1_score, player2_name, player2_score, and winner_name. This table will store each player's name, their score, and the winner's name for each ping pong game.

> Finished chain.
""",
"""
Human $ record the following games that peyton and lasse played. i'll separate them with semicolons: peyton won 21-2; lasse won 21-5; lasse wins 21-10
""",
"""
> Entering new AgentExecutor chain...
Thought: I need to insert the given game records into the ping_pong_games table.
Action: SQL executor plugin
Action Input: INSERT INTO ping_pong_games (player1_name, player1_score, player2_name, player2_score, winner_name) VALUES ('Peyton', 21, 'Lasse', 2, 'Peyton'), ('Peyton', 5, 'Lasse', 21, 'Lasse'), ('Peyton', 10, 'Lasse', 21, 'Lasse');
Observation: Query executed successfully.
""",
"""
Thought:I now know the final answer.
Final Answer: I have recorded the three games between Peyton and Lasse as you provided.

> Finished chain.
""",
"""
Human $ who won more games? what was the total number of points scored in all games?
""",
"""
> Entering new AgentExecutor chain...
Thought: I need to execute a SQL query to find out who won more games and the total number of points scored in all games. I will use the SQL executor plugin to run the query. 

Action: SQL executor plugin
Action Input: SELECT winner_name, COUNT(*) as wins, SUM(player1_score + player2_score) as total_points FROM ping_pong_games GROUP BY winner_name ORDER BY wins DESC;

Observation: ('Lasse', 2, Decimal('57'))
('Peyton', 1, Decimal('23'))
""",
"""
Thought:I now know the final answer
Final Answer: Lasse won more games with 2 wins, while Peyton won 1 game. The total number of points scored in all games is 80.

> Finished chain.
""",
"""
Human $ what does my database schema look like?
""",
"""
> Entering new AgentExecutor chain...
Thought: I need to provide the current database table schema.
Final Answer: Your database schema is as follows:

CREATE TABLE public.ping_pong_games (
    id INT8 NOT NULL DEFAULT unique_rowid(),
    player1_name STRING NOT NULL,
    player1_score INT8 NOT NULL,
    player2_name STRING NOT NULL,
    player2_score INT8 NOT NULL,
    winner_name STRING NOT NULL,
    CONSTRAINT ping_pong_games_pkey PRIMARY KEY (id ASC)
);

> Finished chain.
""",
"""
Human $ wipe my database
""",
"""
> Entering new AgentExecutor chain...
Thought: The user wants to delete the table from the database, so I need to run a DROP TABLE command.

Action: SQL executor plugin
Action Input: DROP TABLE public.ping_pong_games;
Observation: Query executed successfully.
""",
"""
Thought:I now know the final answer.
Final Answer: I have wiped your database by deleting the "ping_pong_games" table.

> Finished chain.
""",
]

cur_msg = 0
def on_key_press(event):
    global cur_msg

    if event.name == 'space':
        print(messages[cur_msg])
        cur_msg += 1
        if cur_msg == len(messages):
            print("Demo finished")

keyboard.on_press(on_key_press)

print("Ready to start demo")

keyboard.wait()