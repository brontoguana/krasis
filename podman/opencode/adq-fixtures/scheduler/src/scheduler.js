export class Scheduler {
  constructor(clock = () => Date.now()) {
    this.clock = clock;
    this.queue = [];
  }

  schedule(id, delayMs) {
    this.queue.push({ id, deadline: this.clock() + delayMs });
    this.queue.sort((a, b) => a.deadline - b.deadline || a.id.localeCompare(b.id));
  }

  due(now = this.clock()) {
    const ready = this.queue.filter((entry) => entry.deadline <= now);
    this.queue = this.queue.filter((entry) => entry.deadline > now);
    return ready.map((entry) => entry.id);
  }
}
