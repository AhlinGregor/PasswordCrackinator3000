package org.example;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Formatter;
import java.util.Stack;

public class SequentialSolution {
    private final static String smallAlpha = "abcdefghijklmnopqrstuvwxyz";
    private final static String bigAlpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    private final static char[] nonAlphabeticalCharacters = {
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  // Digits
            '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',   // Symbols
            '-', '_', '=', '+', '[', ']', '{', '}', '\\', '|',  // Brackets and slashes
            ';', ':', '\'', '\"', ',', '<', '.', '>', '/', '?', // Punctuation
            '`', '~'                                           // Miscellaneous
    };
    private final static String nonAlpha = new String(nonAlphabeticalCharacters);

    // Can I offer you a static method in these trying times

    /**
     * Method to decide if we're computing an MD5 or an SHA-256 hash
     * @param hash Password hash given as a user input
     * @param opt Options for lowercase, uppercase and special characters (including numbers)
     * @param length Length of the password
     * @param progressBar Progress bar component to dynamically update the progress bar while trying hashes
     * @param totalCombinations The total number of combinations possible with the given character set
     * @return  A password that if hashed with the correct algorithm will return the parameter "hash" or null if the password is not found
     */
    public static String computeDizShiz(String hash, int opt, int length, JProgressBar progressBar, long totalCombinations) {
        String available = getCharacterSet(opt);
        if (isValidMD5(hash)) {
            //if (!isValidMD5(hash)) return null;

            // Big daddy method for md5
            return findMatchingPermutationMD(hash, available, length, progressBar, totalCombinations);
        } else if (isValidSHA(hash)) {
            //if (!isValidSHA(hash)) return null;

            // Big daddy method for sha-256
            return findMatchingPermutationSHA(hash, available, length, progressBar, totalCombinations);
        } else {
            return null;
        }
    }

    /**
     * Method to validate if the hash is in line with the SHA-256 requirements
     * @param input User-given hash
     * @return true if it is a valid hash, false otherwise
     */
    private static boolean isValidSHA(String input) {
        return input != null && input.matches("^[a-fA-F0-9]{64}$"); //"Yayy! Regex!" said he, sarcastically
    }

    /**
     * Method that calculates all possible combinations for a given length and charset
     * @param charsetLength All possible characters
     * @param maxLength The password length
     * @return a long integer that represents the number of possible -combinations
     */
    public static long calculateTotalCombinations(int charsetLength, int maxLength) {

        return (long) Math.pow(charsetLength, maxLength);
    }

    /**
     * Method for generating all possible SHA-256 hashes within the given restrictions
     * @param hash User given input
     * @param available String of available characters
     * @param maxLength Length of password
     * @param progressBar Component to update
     * @param totalCombinations Number of total combinations (purely for progress bar functionality
     * @return either cracked password or null if not found
     */
    private static String findMatchingPermutationSHA(String hash, String available, int maxLength, JProgressBar progressBar, long totalCombinations) {
        long[] currentProgress = {0};           // Zaporedno stevilo preizkusov
        Stack<String> stack = new Stack<>();    // A stack of password pancakes
        stack.push("");

        while (!stack.isEmpty()) {
            String prefix = stack.pop();

            // Check if the current permutation has reached the desired length
            if (prefix.length() == maxLength) {
                currentProgress[0]++;           // Update progress
                SwingUtilities.invokeLater(() -> {
                    progressBar.setValue((int) currentProgress[0]);
                    progressBar.setString(currentProgress[0] + "/" + totalCombinations);
                });

                // Dobimo hash
                String computedHash = computeSHA256Hash(prefix);

                if (computedHash.equalsIgnoreCase(hash)) {
                    // naredi zadnji update pbja in vrni rezultat
                    progressBar.setValue((int) currentProgress[0]); // Final update
                    progressBar.setString(currentProgress[0] + "/" + totalCombinations);
                    return prefix;
                }
            }

            // If the prefix length is less than maxLength, generate new permutations
            if (prefix.length() < maxLength) {
                for (int i = 0; i < available.length(); i++) {
                    stack.push(prefix + available.charAt(i));
                }
            }
        }

        // No match found
        return null;
    }

    /**
     * Method for generating all possible MD5 hashes within the given restrictions
     * @param hash User given input
     * @param available String of available characters
     * @param maxLength Length of password
     * @param progressBar Progress bar component to update
     * @param totalCombinations Number of total combinations (purely for progress bar functionality
     * @return either cracked password or null if not found
     */
    private static String findMatchingPermutationMD(String hash, String available, int maxLength, JProgressBar progressBar, long totalCombinations) {
        long[] currentProgress = {0};
        Stack<String> stack = new Stack<>();
        stack.push("");

        while (!stack.isEmpty()) {
            String prefix = stack.pop();

            // Check if the current permutation has reached the desired length
            if (prefix.length() == maxLength) {
                currentProgress[0]++;
                SwingUtilities.invokeLater(() -> {
                    progressBar.setValue((int) currentProgress[0]);
                    progressBar.setString(currentProgress[0] + "/" + totalCombinations);
                });

                String computedHash = computeMD5Hash(prefix);

                if (computedHash.equalsIgnoreCase(hash)) {
                    // Early stopping: Return the matching permutation
                    progressBar.setValue((int) currentProgress[0]); // Final update
                    progressBar.setString(currentProgress[0] + "/" + totalCombinations);
                    return prefix;
                }
            }

            // If the prefix length is less than maxLength, generate new permutations
            if (prefix.length() < maxLength) {
                for (int i = 0; i < available.length(); i++) {
                    stack.push(prefix + available.charAt(i));
                }
            }
        }

        // No match found
        return null;
    }

    /**
     * Method to validate if the hash is in line with the MD5 requirements
     * @param input User-given hash
     * @return true if it is a valid hash, false otherwise
     */
    private static boolean isValidMD5(String input) {
        if (input == null || input.length() != 32) {
            return false;
        }

        return input.matches("[a-fA-F0-9]{32}"); // The bane of my existence yet again
    }

    /**
     * Method to preform a dictionary attack
     * @param file Dictionary file in .txt format
     * @param hash User given hash
     * @return String representing a password or null if password is not in the dictionary file
     */
    public static String dictionaryAttack(File file, String hash) {
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String lineHash;
                if (isValidMD5(hash)) {
                    lineHash = computeMD5Hash(line);
                } else if (isValidSHA(hash)) {
                    lineHash = computeSHA256Hash(line);
                } else {
                    return null;
                }
                if (lineHash.equals(hash)) return line;
            }
        } catch (IOException ex) {
            System.err.println("Error reading the file: " + ex.getMessage());
        }
        return null;
    }

    /**
     * Method to generate the SHA-256 hash
     * @param input String we want to hash
     * @return String representation of a hash
     */
    private static String computeSHA256Hash(String input) {
        MessageDigest md;
        try {
            md = MessageDigest.getInstance("SHA-256");
            byte[] hashBytes = md.digest(input.getBytes());
            return byteArrayToHexString(hashBytes);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Helper method to convert a byte array to a hex String
     * @param bytes Array of bytes we want to convert
     * @return Hex String representation of the array
     */
    private static String byteArrayToHexString(byte[] bytes) {
        Formatter formatter = new Formatter();
        for (byte b : bytes) {
            formatter.format("%02x", b);
        }
        String hexString = formatter.toString();
        formatter.close();
        return hexString;
    }

    /**
     * Method to generate the MD5 hash
     * @param input String we want to hash
     * @return String representation of a hash
     */
    private static String computeMD5Hash(String input) {
        MessageDigest md;
        try {
            md = MessageDigest.getInstance("MD5");
            byte[] hashBytes = md.digest(input.getBytes());
            return byteArrayToHexString(hashBytes);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Method that generates a character set of available character as specified by the user
     * @param opt Integer representation of options
     * @return A String with all possible characters
     */
    public static String getCharacterSet(int opt) {
        return switch (opt) {
            case 1 -> SequentialSolution.smallAlpha;
            case 2 -> SequentialSolution.bigAlpha;
            case 3 -> SequentialSolution.smallAlpha + SequentialSolution.bigAlpha;
            case 4 -> SequentialSolution.nonAlpha;
            case 5 -> SequentialSolution.smallAlpha + SequentialSolution.nonAlpha;
            case 6 -> SequentialSolution.bigAlpha + SequentialSolution.nonAlpha;
            default -> SequentialSolution.smallAlpha + SequentialSolution.bigAlpha + SequentialSolution.nonAlpha;
        };
    }
}
