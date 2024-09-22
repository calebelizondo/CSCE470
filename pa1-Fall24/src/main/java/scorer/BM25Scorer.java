package scorer;

import ds.Document;
import ds.Query;
import utils.IndexUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;

/**
 * Skeleton code for the implementation of a BM25 scorer in Task 2.
 */
public class BM25Scorer extends AScorer {

    /*
     *  TODO: You will want to tune these values
     */
    double titleweight  = 20;
    double bodyweight = 1;

    // BM25-specific weights
    double btitle = 1;
    double bbody = .3;

    double k1 = 5;
    double pageRankLambda = 5;
    double pageRankLambdaPrime = 1;

    // query -> url -> document
    Map<Query,Map<String, Document>> queryDict;

    // BM25 data structures--feel free to modify these
    // ds.Document -> field -> length
    Map<Document,Map<String,Double>> lengths;

    // field name -> average length
    Map<String,Double> avgLengths;

    // ds.Document -> pagerank score
    Map<Document,Double> pagerankScores;

    /**
     * Construct a scorer.BM25Scorer.
     * @param utils Index utilities
     * @param queryDict a map of query to url to document
     */
    public BM25Scorer(IndexUtils utils, Map<Query,Map<String, Document>> queryDict) {
        super(utils);
        this.queryDict = queryDict;
        this.calcAverageLengths();
    }

    /**
     * Set up average lengths for BM25, also handling PageRank.
     */
    public void calcAverageLengths() {
        lengths = new HashMap<>();
        avgLengths = new HashMap<>(); 
        pagerankScores = new HashMap<>();

        /*
         * TODO : Your code here
         * Initialize any data structures needed, perform
         * any preprocessing you would like to do on the fields,
         * accumulate lengths of fields.
         * handle pagerank.  
         */

        for (String tfType : this.TFTYPES) {
            /*
             * TODO : Your code here
             * Normalize lengths to get average lengths for
             * each field (body, title).
             */

            int doc_count = 0;
            double length_total = 0;

            Set<Query> queries = queryDict.keySet();
            for (Query query : queries) {
                for (Document doc : queryDict.get(query).values()) {
                    
                    if (!lengths.containsKey(doc)) lengths.put(doc, new HashMap<>());
                    if (tfType == "title") lengths.get(doc).put(tfType, (double) doc.title_length);
                    else lengths.get(doc).put(tfType, (double) doc.body_length);

                    if (!pagerankScores.containsKey(doc)) pagerankScores.put(doc, (double) doc.page_rank);

                    doc_count += 1;
                    length_total += lengths.get(doc).get(tfType);
                }
            }

            avgLengths.put(tfType, length_total / doc_count);
        }

    }

    /**
     * Get the net score.
     * @param tfs the term frequencies
     * @param q the ds.Query
     * @param tfQuery
     * @param d the ds.Document
     * @return the net score
     */
    public double getNetScore(Map<String,Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery, Document d) {

        double score = 0.0;

        /*
         * TODO : Your code here
         * Use equation 3 first and then equation 4 in the writeup to compute the overall score
         * of a document d for a query q.
         */
        

        // Equation 3
        Set<String> wordSet = new HashSet<String>(q.queryWords); 
        Map<String, Double> wordWeights = new HashMap<>();

        for (String word : wordSet) {
            double weight = 0;
            
            for (String tfType : this.TFTYPES) {
                double weight_param = tfType == "title" ? btitle : bbody;
                weight += weight_param * tfs.get(tfType).get(word);
            }

            wordWeights.put(word, weight);
        }

        //Equation 4
        for (String word : wordSet) {
            double weight = wordWeights.get(word);
            double idf = this.utils.totalNumDocs() / this.utils.docFreq(word);
            double Vj = Math.log(pageRankLambdaPrime + d.page_rank);

            score += (weight * idf) / (k1 + weight) + (pageRankLambda * Vj);
        }

        return score;
    }

    /**
     * Do BM25 Normalization.
     * @param tfs the term frequencies
     * @param d the ds.Document
     * @param q the ds.Query
     */
    public void normalizeTFs(Map<String,Map<String, Double>> tfs, Document d, Query q) {
        /*
         * TODO : Your code here
         * Use equation 2 in the writeup to normalize the raw term frequencies
         * in fields in document d.
         */

         for (String word : tfs.keySet()) {
            for (String tfType : this.TFTYPES) {
                double raw_tf = tfs.get(tfType).getOrDefault(word, 0.0);
                double field_param = (tfType == "title") ? titleweight : bodyweight;
                double field_len = (tfType == "title") ? d.title_length : d.body_length;
                double avg_field_len = avgLengths.get(tfType);
                tfs.get(tfType).replace(word, raw_tf / ((1 - field_param) + (field_param * field_len / avg_field_len)));
            }
         }
    }

    /**
     * Write the tuned parameters of BM25 to file.
     * Only used for grading purpose, you should NOT modify this method.
     * @param filePath the output file path.
     */
    private void writeParaValues(String filePath) {
        try {
            File file = new File(filePath);
            if (!file.exists()) {
                file.createNewFile();
            }
            FileWriter fw = new FileWriter(file.getAbsoluteFile());
            String[] names = {
                    "titleweight", "bodyweight", "btitle",
                    "bbody", "k1", "pageRankLambda", "pageRankLambdaPrime"
            };
            double[] values = {
                    this.titleweight, this.bodyweight, this.btitle,
                    this.bbody, this.k1, this.pageRankLambda,
                    this.pageRankLambdaPrime
            };
            BufferedWriter bw = new BufferedWriter(fw);
            for (int idx = 0; idx < names.length; ++ idx) {
                bw.write(names[idx] + " " + values[idx]);
                bw.newLine();
            }
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    /**
     * Get the similarity score.
     * @param d the ds.Document
     * @param q the ds.Query
     * @return the similarity score
     */
    public double getSimScore(Document d, Query q) {
        Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
        this.normalizeTFs(tfs, d, q);
        Map<String,Double> tfQuery = getQueryFreqs(q);

        // Write out the tuned BM25 parameters
        // This is only used for grading purposes.
        // You should NOT modify the writeParaValues method.
        writeParaValues("bm25Para.txt");
        return getNetScore(tfs,q,tfQuery,d);
    }

}
