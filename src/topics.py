class Topics:
    def __init__(self, query, similar_topics):
        self.query = query
        self.similar_topics = TopicFactory.create_topic_dict(similar_topics)
    
    def update(self, query, similar_topics):
        if query == self.query:
            return False
        else:
            self.query = query
            self.similar_topics = TopicFactory.create_topic_list(similar_topics)
        
    def any_solo(self):
        if any(t.solo for t in self.similar_topics.values()):
            return True
        else:
            False
    
    def get_solo(self):
        return [str(t) for t in self.similar_topics.values() if t.solo]

    def toggle_solo(self, topic_id):
        self.similar_topics[str(topic_id)].toggle_solo()

    def get_selected_topics(self):
        if self.any_solo():
            return self.get_solo()
        return [t.topic_id for t in self.similar_topics.values() if t.selected]

    def select_topic(self, topic_id, choice):
        self.similar_topics[str(topic_id)].select(choice)


class TopicFactory:
    @staticmethod
    def create_topic(topic_id):
        return Topic(topic_id)
    
    @staticmethod
    def create_topic_dict(topic_ids):
        return {str(topic_id):TopicFactory.create_topic(topic_id) for topic_id in topic_ids}

class Topic:
    def __init__(self, topic):
        self.topic_id = str(topic)
        self.selected = True
        self.solo = False
    
    def __str__(self):
        return self.topic_id
    
    def select(self, choice):
        self.selected = choice

    def toggle_solo(self):
        self.solo = not self.solo
    
