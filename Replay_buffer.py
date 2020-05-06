import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ReplayBuffer는 consecutive한 updates들의 상관관계를 줄이기위해서 고안되었습니다.
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, buffer_size, batch_size, priority=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer (chosen as multiple of num agents)
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        # agnet의 step()함수 실행할때, 다음 5가지가 무조건 저장된다.
        self.states = torch.zeros((buffer_size,) + state_size).to(device)
        self.next_states = torch.zeros((buffer_size,) + state_size).to(device)
        self.actions = torch.zeros(buffer_size, 1, dtype=torch.long).to(device)
        self.rewards = torch.zeros(buffer_size, 1, dtype=torch.float).to(device)
        self.dones = torch.zeros(buffer_size, 1, dtype=torch.float).to(device)
        # 그러니까 위에 idx와 e의 idx는 따로논다.(아님 따로 안놈)
        self.e = np.zeros((buffer_size, 1), dtype=np.float)

        self.priority = priority
        # 다음에 저장해야할 메모리 위치를 가리키고 있다.
        self.ptr = 0
        self.n = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.states[self.ptr] = torch.from_numpy(state).to(device)
        self.next_states[self.ptr] = torch.from_numpy(next_state).to(device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        # ReplayBuffer 인스턴스 안에 ptr변수가 0으로 우선 초기화되어있다. add를 한번 할때마다 ptr이 1씩 증가하여
        # 다음 index에 저장하게 해준다.
        self.ptr += 1
        # index가 buffer_size를 초과하면 ptr을 0으로 초기화 시켜 오래된 데이터부터 덮어 씌운다.
        if self.ptr >= self.buffer_size:
            self.ptr = 0
            self.n = self.buffer_size

    def sample(self, get_all=False):
        """Randomly sample a batch of experiences from memory."""
        n = len(self)
        if get_all:
            return self.states[:n], self.actions[:n], self.rewards[:n], self.next_states[:n], self.dones[:n]
        if self.priority:
            # repalce가 true이면 한번 선택한걸 다시 선택 가능합니다.
            # 배열이면 원래의 데이터, 정수이면 arrange(n)명령으로 데이터를 생성합니다.
            # 배열. 각 데이터가 선택될 수 있는 확률
            idx = np.random.choice(n, self.batch_size, replace=False, p=self.e)
        else:
            idx = np.random.choice(n, self.batch_size, replace=False)

        states = self.states[idx]
        next_states = self.next_states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]

        return (states, actions, rewards, next_states, dones), idx

    def update_error(self, e, idx=None):
        # detach()는 텐서에 대하여 기록(history) 추적을 중지하고, 현재의 계산 기록으로 부터 분리시키고 이후에 일어나는 계산들은
        # 추적되지 않게 한다.
        e = torch.abs(e.detach())
        e = (e + 0.01) ** 0.6
        e = e / e.sum()
        # 특정 idx위치에 대해서 update 한다면
        if idx is not None:
            self.e[idx] = e.cpu().numpy()
        # 순서대로 update한다면
        else:
            self.e[:len(self)] = e.cpu().numpy()

    def __len__(self):
        # 현재 ReplayBuffer의 크기를 반환합니다.
        # n이 0일경우 현재 포인터를 반환합니다.
        if self.n == 0:
            return self.ptr
        # n이 0이 아닌 value일 경우 꽉 찾다는 것이므로 n을 반환합니다. 이때 n은 버퍼의 최대크기로 지정되었습니다.
        else:
            return self.n