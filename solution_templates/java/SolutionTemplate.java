/**
 * Box Box Box - F1 Race Simulator Template (Java)
 *
 * This template shows the required input/output structure.
 * Implement your race simulation logic to predict finishing positions.
 */

import com.google.gson.*;
import java.io.*;
import java.util.*;

public class SolutionTemplate {

    public static void main(String[] args) throws IOException {
        // Read test case from stdin
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        Gson gson = new Gson();
        JsonObject testCase = gson.fromJson(reader, JsonObject.class);

        String raceId = testCase.get("race_id").getAsString();
        JsonObject raceConfig = testCase.getAsJsonObject("race_config");
        JsonObject strategies = testCase.getAsJsonObject("strategies");

        // TODO: Implement your race simulation logic here
        // Analyze the historical data in data/historical_races/ to understand
        // how to accurately simulate races and predict finishing positions

        List<String> finishingPositions = new ArrayList<>();  // Replace with your simulation results

        // Output result to stdout
        JsonObject output = new JsonObject();
        output.addProperty("race_id", raceId);

        JsonArray positionsArray = new JsonArray();
        for (String driverId : finishingPositions) {
            positionsArray.add(driverId);
        }
        output.add("finishing_positions", positionsArray);  // Array of 20 driver IDs (1st to 20th)

        System.out.println(gson.toJson(output));
    }
}
