import numpy as np
from scipy.sparse import csr_matrix, vstack

class LSH:
    def __init__(self, num_hashes: int, num_ratings: int) -> None:
        """
        Initialize the LSH class.
        
        :param num_hashes: Number of hash functions.
        :param num_bands: Number of bands to use for hashing.
        """
        self.num_hashes = num_hashes
        self.num_ratings = num_ratings
        self.hash_vecs = [np.random.uniform(-1, 1, num_ratings) for _ in range(num_hashes)]
        self.neighbors_set = {}

    def get_bucket_idx(self, vec):
        hash_input = ''
        for hash_vec in self.hash_vecs:
            dot_product = vec.dot(hash_vec)
            
            if dot_product > 0:
                hash_input += '1'
            else:
                hash_input += '0'
        
        return hash_input

    def insert(self, vec):
        """
        Insert a vector into the LSH hash tables.
        
        :param vec: Vector to insert.
        :return: Hashed value of the input vector
        """
        hash_input = self.get_bucket_idx(vec)
        
        self.insert_to_neighbors(vec, hash_input)
        
        return hash_input
    
    def insert_to_neighbors(self, vec, hash_input):
        vec = csr_matrix(vec)
        if hash_input not in self.neighbors_set:
            self.neighbors_set[hash_input] = vec
        else:
            self.neighbors_set[hash_input] = vstack([self.neighbors_set[hash_input], vec])

    def get_neighbors_set(self, bucket_idx):
        if bucket_idx in self.neighbors_set:
            return self.neighbors_set[bucket_idx].copy()
        else:
            return csr_matrix((0, self.num_ratings))  # Return an empty csr_matrix with the appropriate number of columns

    
    def query(self, vec):
        """
        Query the LSH hash tables for approximate nearest neighbors.
        
        :param vec: Vector to query.
        :return: List of approximate nearest neighbors.
        """
        bucket_idx = self.get_bucket_idx(vec)
        
        neighbors = self.get_neighbors_set(bucket_idx)

        # self.insert_to_neighbors(vec, bucket_idx)
        
        return bucket_idx, neighbors


print(len([{"id": 58217445830, "username": "kingsway_investment_", "full_name": "Asibe Kingsley chisom"}, {"id": 64655367448, "username": "lucidus_yt_", "full_name": "Raziel Davids"}, {"id": 5529088120, "username": "realproudlycoffee", "full_name": "Proudly Coffee"}, {"id": 66912455607, "username": "waveswellnessjourney", "full_name": "Waves Wellness Journey"}, {"id": 51224329751, "username": "alishaukat2cool", "full_name": "Ali Shaukat"}, {"id": 59246864436, "username": "i_was_best_", "full_name": "Drip \ud83d\ude0e"}, {"id": 58951022792, "username": "mohamedelsirag", "full_name": "Mohamed ELsirag"}, {"id": 67399983629, "username": "pseudojail_", "full_name": "PseudoJail"}, {"id": 59575899848, "username": "lokochong801", "full_name": "\u062e\u0648\u0633\u064a\u0647 \u0628\u0627\u062a\u0634\u064a\u0643\u0648"}, {"id": 52134477891, "username": "havocjupiter", "full_name": "David Raser"}, {"id": 530903053, "username": "mt_liberty", "full_name": "MT_liberty"}, {"id": 3927573096, "username": "sam_vvs_", "full_name": "Willy Carter"}, {"id": 258455840, "username": "superleo83", "full_name": "Lucy Wilson"}, {"id": 323763861, "username": "joshchristianau", "full_name": "Josh Christian \ud83c\udde6\ud83c\uddfa"}, {"id": 52933800324, "username": "angie2b2001", "full_name": "Marigolds"}, {"id": 60338176010, "username": "fauzanmarket", "full_name": "Fauzan | Tech | Market | Branding"}, {"id": 20318401387, "username": "p.o_p_l_a", "full_name": "\ubb3c\uace0\uae30"}, {"id": 66627059933, "username": "mariam.260614", "full_name": "mariam"}, {"id": 4588170318, "username": "pawan_pavuluri", "full_name": "p@w@n"}, {"id": 3088272101, "username": "playernum23", "full_name": "Luka"}, {"id": 6649793607, "username": "duyguballiyuce", "full_name": "Duygu Ball\u0131 Y\u00fcce"}, {"id": 6134113847, "username": "amateur_perspect", "full_name": "amateur_perspect"}, {"id": 45411491171, "username": "olamidedeborahhh", "full_name": "Olamide Aniyikaye"}, {"id": 48267009152, "username": "monimax.133", "full_name": "Maxwell Takerhe"}, {"id": 54173124165, "username": "its_nut_jr", "full_name": "Mam N Njie"}, {"id": 51470503183, "username": "ahmadullahashori", "full_name": "Ahmad Ashori"}, {"id": 66088598792, "username": "plungachimangan", "full_name": ""}, {"id": 3064092344, "username": "dooood._", "full_name": "natasha!\ud83c\udf1b\ud83c\udf3c\ud83c\udf1c"}, {"id": 67105285564, "username": "dadslife365", "full_name": "Dads"}, {"id": 67804760568, "username": "the.green.goddesses", "full_name": ""}, {"id": 67660248112, "username": "shreya083039", "full_name": "shreya"}, {"id": 67785762251, "username": "pd_23.07", "full_name": "purvansh"}, {"id": 67944503586, "username": "onetimeofficiall", "full_name": "\u0648\u0627\u0646 \u062a\u0627\u06cc\u0645 | \u0627\u0646\u06af\u06cc\u0632\u0634\u06cc \u2022 \u0645\u0648\u0641\u0642\u06cc\u062a \u2022 \u0644\u0627\u06cc\u0641 \u0627\u0633\u062a\u0627\u06cc\u0644"}, {"id": 6151469369, "username": "akshayxvijay", "full_name": "Akshay Vijay"}, {"id": 40492113466, "username": "salvatoreditommaso_", "full_name": "Salvatore Di Tommaso"}, {"id": 67642169821, "username": "damnoosh0", "full_name": ""}, {"id": 65890608988, "username": "dylan_ft3cap", "full_name": "Dylan Saccone | SMB Funding Savant"}, {"id": 59237036044, "username": "emper_ian", "full_name": "AJ"}, {"id": 60030482047, "username": "tb_property_", "full_name": "Travis Brannan"}, {"id": 68177472339, "username": "jarihna_yjs", "full_name": "\ud835\udc77\ud835\udc8a\ud835\udc8d\ud835\udc82\ud835\udc93 \ud835\udc73\ud835\udc90\ud835\udc93\ud835\udc86\ud835\udc8f\ud835\udc9b\ud835\udc90"}, {"id": 67604060846, "username": "serenaeleon", "full_name": "Serena Eleon"}, {"id": 67993586975, "username": "remeriezdestiny", "full_name": "destiny remeriez"}, {"id": 21622369, "username": "cath.caliman.photos", "full_name": "Cath Caliman Fotografia em Lisboa"}, {"id": 4348567767, "username": "jacob_10119", "full_name": "Jacob Sasse"}, {"id": 1597448024, "username": "dr.monanorozy", "full_name": "\u062f\u06a9\u062a\u0631 \u0645\u0648\u0646\u0627 \u0646\u0648\u0631\u0648\u0632\u06cc"}, {"id": 1770018961, "username": "autumnfladmosmith", "full_name": "Autumn Fladmo Smith"}, {"id": 67649329095, "username": "mylaad97", "full_name": ""}, {"id": 64000791435, "username": "rozh.slemanii", "full_name": "R0zh"}, {"id": 46348862126, "username": "lightlyglimpsed", "full_name": "Linda Corbet"}]))