package scorer;

import ds.Document;
import ds.Query;
import utils.IndexUtils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An abstract class for a scorer.
 * Needs to be extended by each specific implementation of scorers.
 */
public abstract class AScorer {

    // Map: term -> idf
    IndexUtils utils;

    // Various types of term frequencies that you will need
    String[] TFTYPES = {"title","body"};

    /**
     * Construct an abstract scorer with a map of idfs.
     * @param utils index utility functions like map of idf scores
     */
    public AScorer(IndexUtils utils) {
        this.utils = utils;
    }

    /**
     * You can implement your own function to whatever you want for debug string
     * The following is just an example to include page information in the debug string
     * The string will be forced to be 1-line and truncated to only include the first 200 characters
     */
    public String getDebugStr(Document d, Query q)
    {
        return "Pagerank: " + Integer.toString(d.page_rank);
    }

    /**
     * Score each document for each query.
     * @param d the ds.Document
     * @param q the ds.Query
     */
    public abstract double getSimScore(Document d, Query q);

    /**
     * Get frequencies for a query.
     * @param q the query to compute frequencies for
     */
    public Map<String,Double> getQueryFreqs(Query q) {

        // queryWord -> term frequency
        Map<String,Double> tfQuery = new HashMap<>();

        /*
         * TODO : Your code here
         * Compute the raw term frequencies
         * Additionally weight each of the terms using the idf value
         * of the term in the query (we use the provided text corpus to 
         * determine how many documents contain the query terms, which is stored
         * in this.idfs).
         */

        //compute raw frequencies       
        for (String word : q.queryWords) {
            tfQuery.put(word, tfQuery.getOrDefault(word, 0.0) + 1.0);
        }
        
        //weigh by idf
        for (String word : tfQuery.keySet()) {
            long idf = 1;
            if (this.utils.docFreq(word) != 0) idf = this.utils.totalNumDocs() / this.utils.docFreq(word);
            tfQuery.replace(word, tfQuery.getOrDefault(word, 0.0) * idf);
        }

        return tfQuery;
    }


    /*
     * TODO (Optional in case you want to do any preprocessing here) : Your code here
     * Include any initialization and/or parsing methods
     * that you may want to perform on the ds.Document fields
     * prior to accumulating counts.
     * See the ds.Document class in ds.Document.java to see how
     * the various fields are represented.
     */


    /**
     * Accumulate the various kinds of term frequencies
     * for the fields (title, body).
     * You can override this if you'd like, but it's likely
     * that your concrete classes will share this implementation.
     * @param d the ds.Document
     * @param q the ds.Query
     */
    public Map<String,Map<String, Double>> getDocTermFreqs(Document d, Query q) {

        // Map from tf type (field) -> queryWord -> score
        Map<String,Map<String, Double>> tfs = new HashMap<>();

        /*
         * TODO : Your code here
         * Initialize any variables needed
         */

        for (String tfType : TFTYPES) tfs.put(tfType, new HashMap<String, Double>());

        for (String queryWord : q.queryWords) {
            queryWord = queryWord.toLowerCase(); 

            String title = "";
            if (d.title != null) title = d.title.toLowerCase();
            Double title_tf = (double) (title.split(queryWord).length - 1);
            tfs.get("title").put(queryWord, title_tf);

            if (d.body_hits != null && d.body_hits.containsKey(queryWord)) {
                tfs.get("body").put(queryWord, (double) d.body_hits.get(queryWord).size());
            } else {
                tfs.get("body").put(queryWord, 0.0);
            }

            // List<Integer> bodyHits = d.body_hits.get(queryWord);
            // Double body_tf = (bodyHits != null) ? (double) bodyHits.size() : 0.0;
            // tfs.get("body").put(queryWord, body_tf);    
        }

        return tfs;
    }

}
