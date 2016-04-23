package edu.cwru.sepia.agent;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

import java.io.*;
import java.util.*;

public class RLAgent extends Agent {

    /**
     * Set in the constructor. Defines how many learning episodes your agent should run for.
     * When starting an episode. If the count is greater than this value print a message
     * and call sys.exit(0)
     */
    public final int numEpisodes;
    public int currEpisode = 0;
    
    public final int numLearnEps = 10;
    public final int numEvalEps = 5;
    public int currEvalEps = 0;
    public boolean isLearning;
    
    /**
     * List of your footmen and your enemies footmen
     */
    private List<Integer> myFootmen;
    private List<Integer> enemyFootmen;

    /**
     * Convenience variable specifying enemy agent number. Use this whenever referring
     * to the enemy agent. We will make sure it is set to the proper number when testing your code.
     */
    public static final int ENEMY_PLAYERNUM = 1;

    /**
     * Set this to whatever size your feature vector is.
     */
    public static final int NUM_FEATURES = 5;

    /** Use this random number generator for your epsilon exploration. When you submit we will
     * change this seed so make sure that your agent works for more than the default seed.
     */
    public final Random random = new Random(12345);

    /**
     * Your Q-function weights.
     */
    public Double[] weights;
    
    //Track the weights and features from the previous turns
    public Double[] prevWeights;
    public HashMap<Integer, Double[]> prevFeatures; //Map of the features for each footman in the previous turn

    public HashMap<Integer, Action> lastCommands;
    
    /**
     * These variables are set for you according to the assignment definition. You can change them,
     * but it is not recommended. If you do change them please let us know and explain your reasoning for
     * changing them.
     */
    public final double gamma = 0.9;
    public final double learningRate = .0001;
    public final double epsilon = .02;
    
    /**
     * Helper class to represent individual footmen.
     * @author Joe
     *
     */
    public class Footman {
    	public int id, x, y, hp, team;
    	public int lastTarget;
    	public boolean dead = false;
    	public Footman(int id, State.StateView sv, History.HistoryView hv) {
    		this.id = id;
    		this.team = (myFootmen.contains(id)) ? 0 : ENEMY_PLAYERNUM;
    		if(sv.getTurnNumber() > 0) {
	    		//Officially check the deathLogs to confirm death rather than assume that hp will tell you if the unit is dead
	    		List<DeathLog> deathLogs = hv.getDeathLogs(sv.getTurnNumber()-1);
	    		for(DeathLog death : deathLogs) {
	    			if(death.getDeadUnitID() == this.id) {
	    				this.dead = true;
	    			}
	    		}
    		}
    		//Presumably every action here is a composite attack, and thus a Targeted Action
    		TargetedAction lastAction = (TargetedAction) lastCommands.get(id);
    		if(lastAction != null) {
    			this.lastTarget = lastAction.getTargetId();
    		} else { 
    			this.lastTarget = enemyFootmen.get(0); 
    		}
    		if(!dead) {
	    		Unit.UnitView unit = sv.getUnit(id);
	    		this.x = unit.getXPosition();
	    		this.y = unit.getYPosition();
	    		this.hp = unit.getHP();
    		}
    		//If dead, the unit object will return null when trying to retrieve position
    		else {
    			this.x = 0; this.y = 0; this.hp = 0;
    		}
    		
    	}
    	
    	/**
    	 * chebyshevDistance from another Footman
    	 * @param enemy
    	 * @return
    	 */
    	public double chebyshevDistFrom(Footman enemy) {
    		return Math.max(Math.abs(x - enemy.x), Math.abs(y - enemy.y));
    	}
    }

    public RLAgent(int playernum, String[] args) {
        super(playernum);

        if (args.length >= 1) {
            numEpisodes = Integer.parseInt(args[0]);
            System.out.println("Running " + numEpisodes + " episodes.");
        } else {
            numEpisodes = 10;
            System.out.println("Warning! Number of episodes not specified. Defaulting to 10 episodes.");
        }

        boolean loadWeights = false;
        if (args.length >= 2) {
            loadWeights = Boolean.parseBoolean(args[1]);
        } else {
            System.out.println("Warning! Load weights argument not specified. Defaulting to not loading.");
        }

        if (loadWeights) {
            weights = loadWeights();
        } else {
            // initialize weights to random values between -1 and 1
            weights = new Double[NUM_FEATURES];
            for (int i = 0; i < weights.length; i++) {
                weights[i] = random.nextDouble() * 2 - 1;
            }
        }
        
        if(prevFeatures == null) {
        	prevFeatures = new HashMap<>();
        }
        if(lastCommands == null) {
        	lastCommands = new HashMap<>();
        }
    }

    /**
     * We've implemented some setup code for your convenience. Change what you need to.
     */
    @Override
    public Map<Integer, Action> initialStep(State.StateView sv, History.HistoryView hv) {
    	
    	// at the beginning of a turn
        if (sv.getTurnNumber() == 0){handleEpisodeCount();}

        // Find all of your units
        myFootmen = new LinkedList<>();
        for (Integer unitId : sv.getUnitIds(playernum)) {
            Unit.UnitView unit = sv.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                myFootmen.add(unitId);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }

        // Find all of the enemy units
        enemyFootmen = new LinkedList<>();
        for (Integer unitId : sv.getUnitIds(ENEMY_PLAYERNUM)) {
            Unit.UnitView unit = sv.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                enemyFootmen.add(unitId);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }

        return middleStep(sv, hv);
    }

    /**
     * You will need to calculate the reward at each step and update your totals. You will also need to
     * check if an event has occurred. If it has then you will need to update your weights and select a new action.
     *
     * If you are using the footmen vectors you will also need to remove killed units. To do so use the hv
     * to get a DeathLog. Each DeathLog tells you which player's unit died and the unit ID of the dead unit. To get
     * the deaths from the last turn do something similar to the following snippet. Please be aware that on the first
     * turn you should not call this as you will get nothing back.
     *
     * for(DeathLog deathLog : hv.getDeathLogs(stateView.getTurnNumber() -1)) {
     *     System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
     * }
     *
     *
     * @return New actions to execute or nothing if an event has not occurred.
     */
    @Override
    public Map<Integer, Action> middleStep(State.StateView sv, History.HistoryView hv) {
    	
    	HashMap<Integer, Action> sepiaActions = new HashMap<Integer, Action>();
    	
    	if(eventHasOccurred(sv, hv) || sv.getTurnNumber() == 0) {
        	for(Integer f: myFootmen) {
        		double reward = calculateReward(sv, hv, f);
        		//First check if this footman has previous features on record
        		Double[] oldFeatures = prevFeatures.get(f);
        		//If not, give 0 for all features
        		if(oldFeatures == null) {
        			oldFeatures = new Double[]{0.0,0.0,0.0,0.0,0.0};
        		}
        		weights = updateWeights(weights, oldFeatures, reward, sv, hv, f);
        		int newTarget = selectAction(sv, hv, f);
        		sepiaActions.put(f,Action.createCompoundAttack(f, newTarget));
        		//Keep a record of all of the latest commands issued
        		lastCommands.put(f,Action.createCompoundAttack(f, newTarget));
        	}
         }
    	
    	//Don't look for deaths on first turn
    	if(sv.getTurnNumber() > 1) {
    		//Remove all dead units from the unit lists
    		for(DeathLog deathLog : hv.getDeathLogs(sv.getTurnNumber() -1)) {
    			System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
    			if(myFootmen.contains(deathLog.getDeadUnitID())) {
    				myFootmen.remove(deathLog.getDeadUnitID());
    			} else {
    				enemyFootmen.remove(deathLog.getDeadUnitID());
    			}
    		}
    	}
    	
    	return sepiaActions;
    }

    /**
     * Here you will calculate the cumulative average rewards for your testing episodes. If you have just
     * finished a set of test episodes you will call out testEpisode.
     *
     * It is also a good idea to save your weights with the saveWeights function.
     */
    @Override
    public void terminalStep(State.StateView sv, History.HistoryView hv) {

        // MAKE SURE YOU CALL printTestData after you finish a test episode.

        // Save your weights
        saveWeights(weights);
        
        // Save the rest of player data
        savePlayerData(System.out);

    }
    
    /**
     * Determines whether an event has occurred in the last turn
     * 
     * You should also check for completed actions using the history view. Obviously you never want a footman just
     * sitting around doing nothing (the enemy certainly isn't going to stop attacking). So at the minimum you will
     * have an event whenever one your footmen's targets is killed or an action fails. Actions may fail if the target
     * is surrounded or the unit cannot find a path to the unit. To get the action results from the previous turn
     * you can do something similar to the following. Please be aware that on the first turn you should not call this
     *
     * Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
     * for(ActionResult result : actionResults.values()) {
     *     System.out.println(result.toString());
     * }
     * @param sv
     * @param hv
     * @return
     */
    private boolean eventHasOccurred(State.StateView sv, History.HistoryView hv) {
    	Map<Integer, ActionResult> actionResults = hv.getCommandFeedback(playernum, sv.getTurnNumber() - 1);
    	//If one of my footmen has completed an action, he needs a new action
    	for(ActionResult result : actionResults.values()) {
    		if(myFootmen.contains(result.getAction().getUnitId()) && (result.getFeedback() == ActionFeedback.COMPLETED || result.getFeedback() == ActionFeedback.INCOMPLETEMAYBESTUCK)) {
    			return true;
    		}
    	}
    	//Any death, regardless of friend or foe, should be considered an event
    	if(!hv.getDeathLogs(sv.getTurnNumber() -1).isEmpty()) {
    		for(DeathLog deathLog : hv.getDeathLogs(sv.getTurnNumber() -1)) {
    			System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
    		}
    	    return true;
    	}
    	//else 
    	return false;
    }
    
    /**
     * Handles the logic for determining if we are doing a learning or evaluation episode
     */
    private void handleEpisodeCount(){

    	// do we need to do the evaluation episodes
    	if ((currEpisode % numLearnEps == 0) && (currEpisode != 0)){
    		isLearning = false;
    		if (currEvalEps > numEvalEps){
    			currEpisode++;
    			currEvalEps = 0;
    			isLearning = true;
    		}
    		currEvalEps++;
    	} else {
    		isLearning = true;
    		currEpisode++;
    	}
     	
    	System.out.println("Beginning "+ (isLearning?"Learning":"Evaluation") +" Episode "+(isLearning?currEpisode:currEvalEps));
    }

    /**
     * Calculate the updated weights for this agent. 
     * @param oldWeights Weights prior to update
     * @param oldFeatures Features from (s,a)
     * @param totalReward Cumulative discounted reward for this footman.
     * @param sv Current state of the game.
     * @param hv History of the game up until this point
     * @param footmanId The footman we are updating the weights for
     * @return The updated weight vector.
     */
    public Double[] updateWeights(Double[] oldWeights, Double[] oldFeatures, double totalReward, State.StateView sv, History.HistoryView hv, int footmanId) {
    	double reward = calculateReward(sv, hv, footmanId);
    	int bestTarget = selectAction(sv, hv, footmanId);
    	double oldQ = 0.0; //Qw(s,a)
    	for(int i = 0; i < oldWeights.length; i++) {
    		oldQ += oldWeights[i] * oldFeatures[i];
    	}
    	//wi <- wi + alpha * (R(s,a) + gamma * max a' Qw(s',a') - Qw(s,a)) * fi(s,a)
    	for(int i = 0; i < oldWeights.length; i++) {
        	weights[i] = oldWeights[i] + learningRate * 
        			(reward + gamma * calcQValue(sv, hv, footmanId, bestTarget) - oldQ) * oldFeatures[i];
        }
    	return weights;
    }

    /**
     * Given a footman and the current state and history of the game select the enemy that this unit should
     * attack. This is where you would do the epsilon-greedy action selection.
     *
     * @param sv Current state of the game
     * @param hv The entire history of this episode
     * @param attackerId The footman that will be attacking
     * @return The enemy footman ID this unit should attack
     */
    public int selectAction(State.StateView sv, History.HistoryView hv, int attackerId) {
        double maxValue = 0.0;
        int targetId = enemyFootmen.get(0);
        //Attack the enemy which has the highest value associated with it
        for(int enemy: enemyFootmen) {
        	double attackingValue = calcQValue(sv, hv, attackerId, enemy);
        	if(attackingValue > maxValue) {
        		maxValue = attackingValue;
        		targetId = enemy;
        	}
        }
        
        double randomVal = random.nextDouble();
        if(randomVal > epsilon) {
        	return targetId;
        }
        //Else perform random action!
        else {
        	targetId = (int) Math.round(randomVal * enemyFootmen.size());
        	return targetId;
        }
    	
    }
    

    /**
     * Given the current state and the footman in question calculate the reward received on the last turn.
     * This is where you will check for things like Did this footman take or give damage? Did this footman die
     * or kill its enemy. Did this footman start an action on the last turn? See the assignment description
     * for the full list of rewards.
     *
     * Remember that you will need to discount this reward based on the timestep it is received on. See
     * the assignment description for more details.
     *
     * As part of the reward you will need to calculate if any of the units have taken damage. You can use
     * the history view to get a list of damages dealt in the previous turn. Use something like the following.
     *
     * for(DamageLog damageLogs : historyView.getDamageLogs(lastTurnNumber)) {
     *     System.out.println("Defending player: " + damageLog.getDefenderController() + " defending unit: " + \
     *     damageLog.getDefenderID() + " attacking player: " + damageLog.getAttackerController() + \
     *     "attacking unit: " + damageLog.getAttackerID());
     * }
     *
     * You will do something similar for the deaths. See the middle step documentation for a snippet
     * showing how to use the deathLogs.
     *
     * To see if a command was issued you can check the commands issued log.
     *
     * Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
     * for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
     *     System.out.println("Unit " + commandEntry.getKey() + " was command to " + commandEntry.getValue().toString);
     * }
     *
     * @param sv The current state of the game.
     * @param historyView History of the episode up until this turn.
     * @param footmanId The footman ID you are looking for the reward from.
     * @return The current reward
     */
    public double calculateReward(State.StateView sv, History.HistoryView hv, int footmanId) {
    	Footman attacker = new Footman(footmanId, sv, hv);
    	Footman defender = new Footman(attacker.lastTarget, sv, hv);
    	
    	//Killed target?
    	double killedTarget = 0;
    	if(defender.dead) {
    		killedTarget = 30; //Some large number, likely larger than either damage given or taken
    	}
    	
    	//Died?
    	if(attacker.dead) {
    		//Short circuit and return a poor reward. Don't die
    		return -100;
    	}
    	
    	//Damage dealt
    	double damageDealt = 0;
    	for(DamageLog damageLog : hv.getDamageLogs(sv.getTurnNumber()-1)) {
    		if(damageLog.getAttackerID() == attacker.id) {
    			damageDealt = damageLog.getDamage();
    			break;
    		}
    	}
    	
    	//Damage taken
    	double damageTaken = 0;
    	for(DamageLog damageLog : hv.getDamageLogs(sv.getTurnNumber()-1)) {
    		if(damageLog.getDefenderID() == attacker.id) {
    			damageTaken += damageLog.getDamage();
    			//Evidence that the last command issued to this enemy unit was to attack this unit
    			lastCommands.put(defender.id, Action.createCompoundAttack(defender.id, attacker.id));
    		}
    	}
    	
    	//Started action last turn
    	//I guess it's beneficial if this footman just recently started an action?
    	double startedLastTurn = 0;
    	Action action = hv.getCommandsIssued(playernum, sv.getTurnNumber()-1).get(attacker.id);
    	if(action != null) {
    		startedLastTurn = 10;
    	}
    	
    	return killedTarget + damageDealt - damageTaken + startedLastTurn;
    }
    
    private double cumulativeReward(State.StateView sv, History.HistoryView hv) {
    	double totalReward = 0.0;
    	for(Integer f: myFootmen) {
    		totalReward += calculateReward(sv, hv, f);
    	}
    	return totalReward;
    }

    /**
     * Calculate the Q-Value for a given state action pair. The state in this scenario is the current
     * state view and the history of this episode. The action is the attacker and the enemy pair for the
     * SEPIA attack action.
     *
     * This returns the Q-value according to your feature approximation. This is where you will calculate
     * your features and multiply them by your current weights to get the approximate Q-value.
     *
     * @param sv Current SEPIA state
     * @param hv Episode history up to this point in the game
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman that your footman would be attacking
     * @return The approximate Q-value
     */
    public double calcQValue(State.StateView sv,
                             History.HistoryView hv,
                             int attackerId,
                             int defenderId) {
    	double QSum = 0;
    	
    	Double[] featureVector = calculateFeatureVector(sv, hv, attackerId, defenderId);
    	if(featureVector.length != weights.length) {
    		System.err.println(String.format("Error: Different sizes of weights: %i and feature vector: %i",weights.length, featureVector.length));
    	}
    	for(int i = 0; i < featureVector.length; i++) {
    		QSum += weights[i] * featureVector[i];
    	}
        return QSum;
    }

    /**
     * Given a state and action calculate your features here. Please include a comment explaining what features
     * you chose and why you chose them.
     *
     * All of your feature functions should evaluate to a double. Collect all of these into an array. You will
     * take a dot product of this array with the weights array to get a Q-value for a given state action.
     *
     * It is a good idea to make the first value in your array a constant. This just helps remove any offset
     * from 0 in the Q-function. The other features are up to you. Many are suggested in the assignment
     * description.
     *
     * @param stateView Current state of the SEPIA game
     * @param hv History of the game up until this turn
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman. The one you are considering attacking.
     * @return The array of feature function outputs.
     */
    public Double[] calculateFeatureVector(State.StateView sv,
                                           History.HistoryView hv,
                                           int attackerId,
                                           int defenderId) {
    	//First value constant
    	double constant = 1.0;
    	
    	//Calculate distance away
    	Footman attacker = new Footman(attackerId, sv, hv);
    	Footman defender = new Footman(defenderId, sv, hv);
    	double chebyshevDistAway = attacker.chebyshevDistFrom(defender);
    	
    	//Health difference
    	double hpDiff = attacker.hp - defender.hp; 
    	
    	//Calculate the number of other footmen attacking that same target
    	double otherAttackers = 0;
    	for(Map.Entry<Integer, Action> entry : hv.getCommandsIssued(playernum, sv.getTurnNumber()).entrySet()) {
    		TargetedAction action = (TargetedAction) entry.getValue();
    		if(myFootmen.contains(action.getUnitId()) && action.getTargetId() == defenderId) {
    			otherAttackers++;
    		}
    	}
    	
    	//Is defender attacking me? -1 if yes and 1 if no
    	double defenderAttacking = (defender.lastTarget == attackerId) ? -1 : 1;
    	
    	Double[] featureVector = new Double[]{constant, chebyshevDistAway, hpDiff, otherAttackers, defenderAttacking};
    	
    	prevFeatures.put(attackerId, featureVector);
    	
        return featureVector;
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * Prints the learning rate data described in the assignment. Do not modify this method.
     *
     * @param averageRewards List of cumulative average rewards from test episodes.
     */
    public void printTestData (List<Double> averageRewards) {
        System.out.println("");
        System.out.println("Games Played      Average Cumulative Reward");
        System.out.println("-------------     -------------------------");
        for (int i = 0; i < averageRewards.size(); i++) {
            String gamesPlayed = Integer.toString(10*i);
            String averageReward = String.format("%.2f", averageRewards.get(i));

            int numSpaces = "-------------     ".length() - gamesPlayed.length();
            StringBuffer spaceBuffer = new StringBuffer(numSpaces);
            for (int j = 0; j < numSpaces; j++) {
                spaceBuffer.append(" ");
            }
            System.out.println(gamesPlayed + spaceBuffer.toString() + averageReward);
        }
        System.out.println("");
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will take your set of weights and save them to a file. Overwriting whatever file is
     * currently there. You will use this when training your agents. You will include th output of this function
     * from your trained agent with your submission.
     *
     * Look in the agent_weights folder for the output.
     *
     * @param weights Array of weights
     */
    public void saveWeights(Double[] weights) {
        File path = new File("agent_weights/weights.txt");
        // create the directories if they do not already exist
        path.getAbsoluteFile().getParentFile().mkdirs();

        try {
            // open a new file writer. Set append to false
            BufferedWriter writer = new BufferedWriter(new FileWriter(path, false));

            for (double weight : weights) {
                writer.write(String.format("%f\n", weight));
            }
            writer.flush();
            writer.close();
        } catch(IOException ex) {
            System.err.println("Failed to write weights to file. Reason: " + ex.getMessage());
        }
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will load the weights stored at agent_weights/weights.txt. The contents of this file
     * can be created using the saveWeights function. You will use this function if the load weights argument
     * of the agent is set to 1.
     *
     * @return The array of weights
     */
    public Double[] loadWeights() {
        File path = new File("agent_weights/weights.txt");
        if (!path.exists()) {
            System.err.println("Failed to load weights. File does not exist");
            return null;
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            String line;
            List<Double> weights = new LinkedList<>();
            while((line = reader.readLine()) != null) {
                weights.add(Double.parseDouble(line));
            }
            reader.close();

            return weights.toArray(new Double[weights.size()]);
        } catch(IOException ex) {
            System.err.println("Failed to load weights from file. Reason: " + ex.getMessage());
        }
        return null;
    }

    @Override
    public void savePlayerData(OutputStream outputStream) {
   
    }

    @Override
    public void loadPlayerData(InputStream inputStream) {

    }
}
