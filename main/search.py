# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in search_agents.py).
"""

import os
import random
from builtins import object
from concurrent.futures.thread import ThreadPoolExecutor
from queue import Queue

# from concurrent.futures import Future, ThreadPoolExecutor
import util


def tiny_maze_search(problem):
	"""
	Returns a sequence of moves that solves tiny_maze.  For any other maze, the
	sequence of moves will be incorrect, so only use this for tiny_maze.
	"""
	from game import Directions

	s = Directions.SOUTH
	w = Directions.WEST
	return [s, s, w, s, w, w, s, w]


def depth_first_search(problem):
	to_visit = util.Stack()
	discovered = set()
	parents = {}

	start = problem.get_start_state()
	discovered.add(start)
	to_visit.push(start)

	while not to_visit.is_empty():
		visiting = to_visit.pop()
		if problem.is_goal_state(visiting):
			path = []
			node = visiting
			while node != start:
				parent, action = parents[node]
				path.append(action)
				node = parent
			path.reverse()
			return path

		for neighbour, action, cost in problem.get_successors(visiting):
			if neighbour not in discovered:
				discovered.add(neighbour)
				to_visit.push(neighbour)
				parents[neighbour] = (visiting, action)
	return []


def breadth_first_search(problem):
	"""Search the shallowest nodes in the search tree first."""
	'*** YOUR CODE HERE ***'
	to_visit = util.Queue()
	discovered = set()
	parents = {}

	start = problem.get_start_state()
	discovered.add(start)
	to_visit.push(start)

	while not to_visit.is_empty():
		visiting = to_visit.pop()

		if problem.is_goal_state(visiting):
			path = []
			node = visiting
			while node != start:
				parent, action = parents[node]
				path.append(action)
				node = parent
			path.reverse()
			return path

		for neighbour, action, cost in problem.get_successors(visiting):
			if neighbour not in discovered:
				discovered.add(neighbour)
				to_visit.push(neighbour)
				parents[neighbour] = (visiting, action)
	return []


def uniform_cost_search(problem, heuristic=None):
	"""Search the node of least total cost first."""
	'*** YOUR CODE HERE ***'
	to_visit = util.PriorityQueue()
	visited = set()
	cost_to = {}
	parents = {}

	start = problem.get_start_state()
	cost_to[start] = 0
	to_visit.push(start, 0)

	while not to_visit.is_empty():
		visiting = to_visit.pop()

		if visiting in visited:
			continue
		visited.add(visiting)

		if problem.is_goal_state(visiting):
			path = []
			node = visiting
			while node != start:
				parent, action = parents[node]
				path.append(action)
				node = parent
			path.reverse()
			return path

		for neighbour, action, cost in problem.get_successors(visiting):
			new_cost = cost_to[visiting] + cost
			if neighbour not in cost_to or new_cost < cost_to[neighbour]:
				parents[neighbour] = (visiting, action)
				cost_to[neighbour] = new_cost
				to_visit.push(neighbour, new_cost)


#
# heuristics
#
def a_really_really_bad_heuristic(position, problem):
	from random import random

	return int(random() * 1000)


def null_heuristic(state, problem=None):
	return 0


def your_heuristic(state, problem=None):
	from search_agents import FoodSearchProblem

	#
	# heuristic for the find-the-goal problem
	#
	if isinstance(problem, SearchProblem):
		# data
		pacman_x, pacman_y = state
		goal_x, goal_y = problem.goal

		# YOUR CODE HERE (set value of optimisitic_number_of_steps_to_goal)

		optimisitic_number_of_steps_to_goal = 0
		return optimisitic_number_of_steps_to_goal
	#
	# traveling-salesman problem (collect multiple food pellets)
	#
	elif isinstance(problem, FoodSearchProblem):
		# the state includes a grid of where the food is (problem isn't ter)
		position, food_grid = state
		pacman_x, pacman_y = position

		# YOUR CODE HERE (set value of optimisitic_number_of_steps_to_goal)

		optimisitic_number_of_steps_to_goal = 0
		return optimisitic_number_of_steps_to_goal


def a_star_search(problem, heuristic=your_heuristic):
	to_visit = util.PriorityQueue()
	visited = set()
	g_cost = {}
	parents = {}

	start = problem.get_start_state()
	g_cost[start] = 0
	to_visit.push(start, 0 + heuristic(start, problem))

	while not to_visit.is_empty():
		visiting = to_visit.pop()

		if visiting in visited:
			continue
		visited.add(visiting)

		if problem.is_goal_state(visiting):
			path = []
			node = visiting
			while node != start:
				parent, action = parents[node]
				path.append(action)
				node = parent
			path.reverse()
			return path

		for neighbour, action, cost in problem.get_successors(visiting):
			new_g = g_cost[visiting] + cost
			if neighbour not in g_cost or new_g < g_cost[neighbour]:
				parents[neighbour] = (visiting, action)
				g_cost[neighbour] = new_g
				to_visit.push(neighbour, new_g + heuristic(neighbour, problem))
	return []


task_queue = Queue()


def safe_call(func, *args):
	result_queue = Queue()
	task_queue.put((func, args, result_queue))
	result = result_queue.get()
	if isinstance(result, Exception):
		raise result
	return result


def process_tasks():
	while not task_queue.empty():
		func, args, result_queue = task_queue.get()
		try:
			result = func(*args)
			result_queue.put(result)
		except Exception as e:
			result_queue.put(e)
		task_queue.task_done()


class Monkey:
	def __init__(self, problem, seed=None) -> None:
		self.problem = problem
		self.current = problem.get_start_state()
		self.previous = None
		self.path = []
		self.rand = random.Random(seed)

	def _pick(self, neighbours):
		if len(neighbours) == 1:
			return neighbours[0]
		else:
			return self.rand.choice(
				[n for n in neighbours if n.state != self.previous]
			)

	def step(self):
		if self.problem.is_goal_state(self.current):
			return self.path
		neighbours = self.problem.get_successors(self.current)
		next = self._pick(neighbours)
		self.path.append(next.action)
		self.previous = self.current
		self.current = next.state


def monkey_search_sync(problem, monkey_count=1, max_moves=0):
	monkeys = [Monkey(problem) for _ in range(monkey_count)]

	iteration = 0
	while max_moves == 0 or iteration <= max_moves:
		for monkey in monkeys:
			path = monkey.step()
			if path:
				print(f'Path found! Length: {len(path)}')
				return path
		iteration += 1
	return []


# Still not working, implement Monkey() anyway maybe
def _monkey_thread(problem, start, max_moves):
	rand = random.Random()
	path = []
	current = start
	previous = None
	while max_moves == 0 or len(path) < max_moves:
		if safe_call(problem.is_goal_state, current):
			print(f'path found of length {len(path)}')
			return path
		neighbours = safe_call(problem.get_successors, current)
		if len(neighbours) == 1:
			next = neighbours[0]
		else:
			next = rand.choice([n for n in neighbours if n.state != previous])
		path.append(next.action)
		previous = current
		current = next.state
	return None


# Still not working
def monkey_search_threaded(problem, monkey_count=10, max_moves=0):
	start = problem.get_start_state()

	with ThreadPoolExecutor(max_workers=monkey_count) as executor:
		monkeys = [
			executor.submit(_monkey_thread, problem, start, max_moves)
			for _ in range(monkey_count)
		]

		while monkeys:
			process_tasks()
			done = []
			for i, monkey in enumerate(monkeys):
				if monkey.done():
					path = monkey.result()
					if path:
						print('got path')
						for m in monkeys:
							m.cancel()
						return path
					else:
						done.append(i)
			if done:
				monkeys = [m for i, m in enumerate(monkeys) if i not in done]
	return []


# (you can ignore this, although it might be helpful to know about)
# This is effectively an abstract class
# it should give you an idea of what methods will be available on problem-objects
class SearchProblem(object):
	"""
	This class outlines the structure of a search problem, but doesn't implement
	any of the methods (in object-oriented terminology: an abstract class).

	You do not need to change anything in this class, ever.
	"""

	def get_start_state(self):
		"""
		Returns the start state for the search problem.
		"""
		util.raise_not_defined()

	def is_goal_state(self, state):
		"""
		  state: Search state

		Returns True if and only if the state is a valid goal state.
		"""
		util.raise_not_defined()

	def get_successors(self, state):
		"""
		  state: Search state

		For a given state, this should return a list of triples, (successor,
		action, step_cost), where 'successor' is a successor to the current
		state, 'action' is the action required to get there, and 'step_cost' is
		the incremental cost of expanding to that successor.
		"""
		util.raise_not_defined()

	def get_cost_of_actions(self, actions):
		"""
		 actions: A list of actions to take

		This method returns the total cost of a particular sequence of actions.
		The sequence must be composed of legal moves.
		"""
		util.raise_not_defined()


if os.path.exists('./hidden/search.py'):
	from hidden.search import *
# fallback on a_star_search
for function in [
	breadth_first_search,
	depth_first_search,
	uniform_cost_search,
]:
	try:
		function(None)
	except util.NotDefined:
		exec(f'{function.__name__} = a_star_search', globals(), globals())
	except:
		pass

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
mss = monkey_search_sync
mst = monkey_search_threaded
