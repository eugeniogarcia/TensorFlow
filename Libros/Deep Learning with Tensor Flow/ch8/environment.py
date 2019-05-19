import copy
import random
import shutil
import numpy as np
import tensorflow as tf
import deepchem as dc
import collections

class Environment(object):
  """An environment in which an actor performs actions to accomplish a task.

  An environment has a current state, which is represented as either a single NumPy
  array, or optionally a list of NumPy arrays.  When an action is taken, that causes
  the state to be updated.  Exactly what is meant by an "action" is defined by each
  subclass.  As far as this interface is concerned, it is simply an arbitrary object.
  The environment also computes a reward for each action, and reports when the task
  has been terminated (meaning that no more actions may be taken).
  """

  def __init__(self, state_shape, n_actions, state_dtype=None):
    """Subclasses should call the superclass constructor in addition to doing their own initialization."""
    #dimensiones del estado
    self.state_shape = state_shape
    #Numero de acciones
    self.n_actions = n_actions
    #Asumimos que el estado es float32
    if state_dtype is None:
      # Assume all arrays are float32.
      if isinstance(state_shape[0], collections.Sequence):
        self.state_dtype = [np.float32] * len(state_shape)
      else:
        self.state_dtype = np.float32
    else:
      self.state_dtype = state_dtype


class TicTacToeEnvironment(Environment):
  """
  Play tictactoe against a randomly acting opponent
  """
  #En cada celdilla del tablero podemos encontrarnos uno de estos tres
  #valores: O, X o EMPTY
  #Representa un movimiento de X. Es de dimension 2
  X = np.array([1.0, 0.0])
  #Representa un movimiento de O. Es de dimension 2
  O = np.array([0.0, 1.0])
  #La celdilla esta sin informar
  EMPTY = np.array([0.0, 0.0])

  #Rewards
  ILLEGAL_MOVE_PENALTY = -3.0
  LOSS_PENALTY = -3.0
  NOT_LOSS = 0.1
  DRAW_REWARD = 5.0
  WIN_REWARD = 10.0

  def __init__(self):
    #Llama al constructor de la clase padre
    #EL tablero es 3x3. En el rango tres tenemos una dimension 2. Esto es porque los 
    #valores que vamos a guardar en cada posicion del tablero es de dimension 2
    #Lo que guardaremos es bien O, bien X, bien EMPTY. Estos tres valores los hemos definido
    #al principio de la clase
    #Hay 9 acciones posibles
    super(TicTacToeEnvironment, self).__init__([(3, 3, 2)], 9)
   
    #Define algunas propiedades en el entorno
    self.state = None
    self.terminated = None
    self.reset()


  #Hace el primer movimiento
  #retorna una dupla al azar con las coordinadas del tablero
  #Las posiciones que son candidatas a ser elegidas son aquellas EMPTY
  def get_O_move(self):
    #Crea una lista vacia...
    empty_squares = []
    for row in range(3):
      for col in range(3):
        #Si la posicion del tablero esta EMPTY...
        if np.all(self.state[0][row][col] == TicTacToeEnvironment.EMPTY):
          #... añade la tupla en la lista
          empty_squares.append((row, col))
    #... Ahora elige al azar una dupla de la lista
    return random.choice(empty_squares)

  def reset(self):
    self.terminated = False
    #Limpia el tablero. Basicamente en cada posicion tendremos EMPTY
    #El estado es una lista de largo uno, y en esa posicion tenemos un array de rango 3
    con dimentiones, 3, 3 y 2
    self.state = [np.zeros(shape=(3, 3, 2), dtype=np.float32)]

    # Randomize who goes first
    #Elige si salen X o O
    if random.randint(0, 1) == 1:
      #Toma una posicion al azar del tablero, que este vacia
      move = self.get_O_move()
      #Pone una O en la pisicion move
      self.state[0][move[0]][move[1]] = TicTacToeEnvironment.O

  #Comprueba si hay un ganador
  def check_winner(self, player):
    for i in range(3):
      #Suma los valores en todas las columnas del tablero. Al especificar axis 0 
      #axis 0 suma en columnas
      #axis 1 suma en filas
      #lo que hacemos es obtner dos valores que suman el contenido de cada tupla del tablero
      #en columnas. cada valor representara a un jugador. La primera a X, la sugunda a O
      row = np.sum(self.state[0][i][:], axis=0)
      if np.all(row == player * 3):
        return True
      col = np.sum(self.state[0][:][i], axis=0)
      if np.all(col == player * 3):
        return True

    diag1 = self.state[0][0][0] + self.state[0][1][1] + self.state[0][2][2]
    if np.all(diag1 == player * 3):
      return True
    diag2 = self.state[0][0][2] + self.state[0][1][1] + self.state[0][2][0]
    if np.all(diag2 == player * 3):
      return True
    return False

  #Si todas las celdas estan informadas, termino el juego
  def game_over(self):
    for i in range(3):
      for j in range(3):
        if np.all(self.state[0][i][j] == TicTacToeEnvironment.EMPTY):
          return False
    return True

  # Se trata de la funcion que clacula el Reward en funcion de la accion, y el estado  
  def step(self, action):
    self.state = copy.deepcopy(self.state)
    #La accion representa una posicion del tablero. Vamos a determinar cual es:
    #Cociente entero
    row = action // 3
    #Resto
    col = action % 3

    #Si la posicion estaba ocupada, es un movimiento ilegal
    #Devolvemos una penalizacion
    if not np.all(self.state[0][row][col] == TicTacToeEnvironment.EMPTY):
      self.terminated = True
      return TicTacToeEnvironment.ILLEGAL_MOVE_PENALTY

    # Ponemos la ficha X en la posicion
    self.state[0][row][col] = TicTacToeEnvironment.X

    # Ha ganado X?. Si es asi, lo recompesamos...
    if self.check_winner(TicTacToeEnvironment.X):
      self.terminated = True
      return TicTacToeEnvironment.WIN_REWARD

    #Sino ha ganado, pero ha terminado la partida, la recompensa corresponde a juego empatado
    if self.game_over():
      self.terminated = True
      return TicTacToeEnvironment.DRAW_REWARD

    #Si no hemos hanado, y tampoco ha terminado la partida, le toca a la maquina, esto es O mueve
    move = self.get_O_move()
    #Pone la ficha de O en una celda al azar
    self.state[0][move[0]][move[1]] = TicTacToeEnvironment.O

    #Si O Gana, devolvemos la penalizacion...
    if self.check_winner(TicTacToeEnvironment.O):
      self.terminated = True
      return TicTacToeEnvironment.LOSS_PENALTY

    #Si la partida ha terminado la partida, la recompensa corresponde a juego empatado
    if self.game_over():
      self.terminated = True
      return TicTacToeEnvironment.DRAW_REWARD

    #En caso contrario, ni hemos ganado, ni O ha ganado, ni la partida ha terminado, el juego sigue
    return TicTacToeEnvironment.NOT_LOSS

  #Representa visualmente el estado del tablero  
  def display(self):
    state = self.state[0]
    s = ""
    for row in range(3):
      for col in range(3):
        if np.all(state[row][col] == TicTacToeEnvironment.EMPTY):
          s += "_"
        if np.all(state[row][col] == TicTacToeEnvironment.X):
          s += "X"
        if np.all(state[row][col] == TicTacToeEnvironment.O):
          s += "O"
      s += "\n"
    return s
