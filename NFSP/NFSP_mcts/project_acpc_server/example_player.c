/*
Copyright (C) 2011 by the Computer Poker Research Group, University of Alberta
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "game.h"
#include "rng.h"
#include "net.h"


int main(int argc, char **argv)
{
	int sock, len, r, a;
	int32_t min, max;
	uint16_t port;
	double p;
	Game *game;
	MatchState state;
	Action action;
	FILE *file, *toServer, *fromServer;
	struct timeval tv;
	double probs[NUM_ACTION_TYPES];
	double actionProbs[NUM_ACTION_TYPES];
	rng_state_t rng;
	char line[MAX_LINE_LEN];
	char new_line[400];


	/* we make some assumptions about the actions - check them here */
	assert(NUM_ACTION_TYPES == 3);

	if (argc < 4)
	{

		fprintf(stderr, "usage: player game server port\n");
		exit(EXIT_FAILURE);
	}

	/* Define the probabilities of actions for the player */
	probs[a_fold] = 0.06;
	probs[a_call] = (1.0 - probs[a_fold]) * 0.5;
	probs[a_raise] = (1.0 - probs[a_fold]) * 0.5;

	/* Initialize the player's random number state using time */
	gettimeofday(&tv, NULL);
	init_genrand(&rng, tv.tv_usec);

	/* get the game */
	file = fopen(argv[1], "r");
	if (file == NULL)
	{

		fprintf(stderr, "ERROR: could not open game %s\n", argv[1]);
		exit(EXIT_FAILURE);
	}
	game = readGame(file);
	if (game == NULL)
	{

		fprintf(stderr, "ERROR: could not read game %s\n", argv[1]);
		exit(EXIT_FAILURE);
	}
	fclose(file);

	/* connect to the dealer */
	if (sscanf(argv[3], "%"SCNu16, &port) < 1)
	{

		fprintf(stderr, "ERROR: invalid port %s\n", argv[3]);
		exit(EXIT_FAILURE);
	}
	sock = connectTo(argv[2], port);
	if (sock < 0)
	{

		exit(EXIT_FAILURE);
	}
	toServer = fdopen(sock, "w");
	fromServer = fdopen(sock, "r");
	if (toServer == NULL || fromServer == NULL)
	{

		fprintf(stderr, "ERROR: could not get socket streams\n");
		exit(EXIT_FAILURE);
	}

	/* send version string to dealer */
	if (fprintf(toServer, "VERSION:%"PRIu32".%"PRIu32".%"PRIu32"\n",
		VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION) != 14)
	{

		fprintf(stderr, "ERROR: could not get send version to server\n");
		exit(EXIT_FAILURE);
	}
	fflush(toServer);


	int player_port = 8001;
	if (strcmp(argv[4], "Alice") == 0)
	{
		player_port = 8000;
	}

	int fd, new_fd, struct_len, numbytes;
	struct sockaddr_in server_addr;
	struct sockaddr_in client_addr;
	//char buff[BUFSIZ];

	server_addr.sin_family = AF_INET;
	server_addr.sin_port = htons(player_port);          ////
	server_addr.sin_addr.s_addr = INADDR_ANY;
	//bzero(&(server_addr.sin_zero), 8);
	struct_len = sizeof(struct sockaddr_in);

	fd = socket(AF_INET, SOCK_STREAM, 0);
	while (bind(fd, (struct sockaddr *)&server_addr, struct_len) == -1);
	printf("Bind Success!\n");
	while (listen(fd, 10) == -1);
	printf("Listening....\n");
	printf("Ready for Accept,Waitting...\n");
	new_fd = accept(fd, (struct sockaddr *)&client_addr, &struct_len);
	printf("Get the Client.\n");

	double value[MAX_PLAYERS];



	/* play the game! */
	while (fgets(line, MAX_LINE_LEN, fromServer))
	{

		fprintf(stderr, "dealer返回的%s", line);
		/* ignore comments */
		if (line[0] == '#' || line[0] == ';')
		{
			continue;
		}

		if (line[0] == 'Q')
		{
			len = readMatchState(line + 1, game, &state);
			len++;
			fprintf(stderr, "TEST 10\n");
		}
		else
		{
			len = readMatchState(line, game, &state);                          ////
		}

		if (len < 0)
		{

			fprintf(stderr, "ERROR: could not read state %s", line);
			exit(EXIT_FAILURE);
		}


		if (stateFinished(&state.state))    // 计算得分，然后直接发给player，不需再得到回应，也不需反馈给dealer
		{
			/* ignore the game over message */
			if (line[0] == 'Q')
			{
				fprintf(stderr, "TEST 11\n");
				int str_size = strlen(line);
				//需要-1
				if (state.state.holeCards[state.viewingPlayer][0] != (line[str_size - 13] - '0') * 10 + (line[str_size - 12] - '0') - 1)
				{
					state.state.holeCards[state.viewingPlayer][0] = (line[str_size - 7] - '0') * 10 + (line[str_size - 6] - '0') - 1;
					state.state.holeCards[state.viewingPlayer][1] = (line[str_size - 4] - '0') * 10 + (line[str_size - 3] - '0') - 1;
					// python返回的card顺序定义与game.h中一致
				}
			}

			value[state.viewingPlayer] = valueOfState(game, &state.state, state.viewingPlayer);

			line[len++] = ':';
			len += sprintf(&line[len], "%lf", value[state.viewingPlayer]);        // 这句没问题
			line[len++] = '%';

			
			if (write(new_fd, line, len) != len)
			{
				/* couldn't send the line */

				fprintf(stderr, "ERROR: could not send line to player\n");
				return -1;
			}
			fprintf(stderr, "TEST 12\n");
			if (line[0] == 'Q')
			{
				goto continue_query;
			}

			/*9月27日改动*/
			if ((len = read(new_fd, line, MAX_LINE_LEN)) <= 0)                 //////
			{
				fprintf(stderr, "ERROR: could not recv line from player\n");
				return -1;
			}
			/*9月27日改动*/

			continue;
		}

		if (currentPlayer(game, &state.state) != state.viewingPlayer)         // '.'的运算符优先级>'&'
		{
			/* we're not acting */
			fprintf(stderr, "不一致，退出\n");
			continue;
		}


		////
		////



		/* add a colon (guaranteed to fit because we read a new-line in fgets) */
		line[len] = ':';                          // 
		++len;

		/* build the set of valid actions */
		p = 0;
		for (a = 0; a < NUM_ACTION_TYPES; ++a)
		{

			actionProbs[a] = 0.0;
		}

		/* consider fold */
		action.type = a_fold;
		action.size = 0;
		if (isValidAction(game, &state.state, 0, &action))
		{

			actionProbs[a_fold] = probs[a_fold];
			p += probs[a_fold];
		}

		/* consider call */
		action.type = a_call;
		action.size = 0;
		actionProbs[a_call] = probs[a_call];
		p += probs[a_call];

		/* consider raise */
		if (raiseIsValid(game, &state.state, &min, &max))
		{

			actionProbs[a_raise] = probs[a_raise];
			p += probs[a_raise];
		}

		/* normalise the probabilities  */
		assert(p > 0.0);
		for (a = 0; a < NUM_ACTION_TYPES; ++a)
		{

			actionProbs[a] /= p;
		}

		/* choose one of the valid actions at random */
		p = genrand_real2(&rng);
		for (a = 0; a < NUM_ACTION_TYPES - 1; ++a)
		{

			if (p <= actionProbs[a])
			{

				break;
			}
			p -= actionProbs[a];
		}
		action.type = (enum ActionType)a;
		if (a == a_raise)
		{

			action.size = min + genrand_int32(&rng) % (max - min + 1);
		}

		/* do the action! */
		assert(isValidAction(game, &state.state, 0, &action));
		r = printAction(game, &action, MAX_LINE_LEN - len - 2,              ////
			&line[len]);
		if (r < 0)
		{

			fprintf(stderr, "ERROR: line too long after printing action\n");
			exit(EXIT_FAILURE);
		}
		len += r;
		line[len] = '\r';
		++len;
		line[len] = '\n';
		++len;

		// 加入state信息
		len += sprintf(&line[len], "%d", state.state.round);
		for (int i = 0; i <= state.state.round; i++)
		{
			line[len++] = '.';
			len += sprintf(&line[len], "%d", state.state.numActions[i]);
			for (int j = 0; j < state.state.numActions[i]; j++)
			{
				line[len++] = ',';

				if (state.state.action[i][j].type == a_fold)
				{
					int three = 3;
					len += sprintf(&line[len], "%d", three);             // fold:3     将传过来的0改为3
				}
				else
				{
					len += sprintf(&line[len], "%d", state.state.action[i][j].type);
				}
				len += sprintf(&line[len], "%d", state.state.actingPlayer[i][j]);
			}
		}
		// 加入state信息


		//fprintf(stderr, "len:%d\n", len);                     ###
		if (write(new_fd, line, len) != len)
		{
			/* couldn't send the line */

			fprintf(stderr, "ERROR: could not send line to player\n");
			return -1;
		}
continue_query:
		
		memset(new_line, 0, 400);

		fprintf(stderr, "TEST 1\n");
		if ((len = read(new_fd, new_line, 400)) <= 0)                 //////
		{
			fprintf(stderr, "ERROR: could not recv line from player\n");
			return -1;
		}
		fprintf(stderr, "python传来的%s\n", new_line);
		//fprintf(stderr, "len:%d\n", len);                      ###

		// 收到的line以'\n'结束？https://blog.csdn.net/poetteaes/article/details/80160562 

		/////////////////

		fprintf(stderr, "TEST 2\n");
		if (fwrite(new_line, 1, len, toServer) != len)
		{

			fprintf(stderr, "ERROR: could not get send response to server\n");
			exit(EXIT_FAILURE);
		}
		fflush(toServer);
	}

	close(new_fd);
	close(fd);

	return EXIT_SUCCESS;
}

