import logging
import time
import os

pathos = True
try:
    from pathos.multiprocessing import ProcessPool as Pool
except ImportError:
    print('You could install pathos to enable parallel simulation.')
    pathos = False

logger = logging.getLogger(__name__)


class SimObj(object):
    def __init__(self,
                 env,
                 controller,
                 sim_time,
                 animate=True,
                 path=None):
        self.env = env
        self.controller = controller
        self.sim_time = sim_time
        self.animate = animate
        self._ctrller_kwargs = None
        self.path = path

    def simulate(self):
        self.controller.reset()
        obs, reward, done, info = self.env.reset()
        tic = time.time()
        while self.env.time < self.env.scenario.start_time + self.sim_time:
            if self.animate:
                self.env.render()
            action = self.controller.policy(obs, reward, done, **info)
            obs, reward, done, info = self.env.step(action)
        toc = time.time()
        logger.info('Simulation took {} seconds.'.format(toc - tic))

    def results(self):
        df = self.env.show_history()
        # Optional: controllers may expose a debug dataframe indexed by Time.
        # This allows adding controller-side signals (e.g., event-trigger diagnostics)
        # into the saved CSV without changing the env/action interface.
        get_dbg = getattr(self.controller, "get_debug_df", None)
        if callable(get_dbg):
            try:
                dbg_df = get_dbg()
                # Join on Time index; keep original rows intact.
                if dbg_df is not None and getattr(dbg_df, "empty", True) is False:
                    df = df.join(dbg_df, how="left")
            except Exception:
                # Never break simulation due to optional logging.
                pass
        return df

    def save_results(self):
        df = self.results()
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        filename = os.path.join(self.path, str(self.env.patient.name) + '.csv')
        df.to_csv(filename)

    def reset(self):
        self.env.reset()
        self.controller.reset()


def sim(sim_object):
    print("Process ID: {}".format(os.getpid()))
    print('Simulation starts ...')
    sim_object.simulate()
    sim_object.save_results()
    print('Simulation Completed!')
    return sim_object.results()


def batch_sim(sim_instances, parallel=False):
    tic = time.time()
    if parallel and pathos:
        with Pool() as p:
            results = p.map(sim, sim_instances)
    else:
        if parallel and not pathos:
            print('Simulation is using single process even though parallel=True.')
        results = [sim(s) for s in sim_instances]
    toc = time.time()
    print('Simulation took {} sec.'.format(toc - tic))
    return results
