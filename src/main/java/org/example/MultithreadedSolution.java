package org.example;
import javax.swing.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Formatter;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public class MultithreadedSolution {
    private final static String smallAlpha = "zyxwvutsrqponmlkjihgfedcba";
    private final static String bigAlpha = "ZYXWVUTSRQPONMLKJIHGFEDCBA";
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
        final boolean[] stopRequested = {false};
        AtomicLong currentProgress = new AtomicLong(0);
        AtomicReference<String> result = new AtomicReference<>(null);
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        AtomicLong lastUIUpdateTime = new AtomicLong(System.currentTimeMillis());
        final int UI_UPDATE_STEP = 1000; // update every 1000 steps or 100ms

        // BlockingQueue<String> queue = new LinkedBlockingQueue<>();
        LinkedBlockingDeque<String> stack = new LinkedBlockingDeque<>();
        stack.push(""); // Start with empty prefix

        AtomicInteger activeTasks = new AtomicInteger(1); // Start with 1 for the initial ""
        int threadCount = Runtime.getRuntime().availableProcessors();

        Runnable worker = () -> {
            try {
                while (!stopRequested[0]) {
                    if (result.get() != null && activeTasks.get() == 0) return;

                    String prefix = stack.take();
                    // if (prefix == null) {
                    //     if (activeTasks.get() == 0) return;
                    //     continue;
                    // }

                    if (prefix.length() == maxLength) {
                        String computedHash = computeSHA256Hash(prefix);
                        long progress = currentProgress.incrementAndGet();
                        long now = System.currentTimeMillis();

                        if (progress % UI_UPDATE_STEP == 0 || (now - lastUIUpdateTime.get()) > 100) {
                            lastUIUpdateTime.set(now);
                            SwingUtilities.invokeLater(() -> {
                                progressBar.setValue((int) progress);
                                progressBar.setString(progress + "/" + totalCombinations);
                            });
                        }


                        if (computedHash.equalsIgnoreCase(hash)) {
                            result.compareAndSet(null, prefix);
                            stopRequested[0] = true;
                        }

                    } else {
                        for (int i = 0; i < available.length(); i++) {
                            String newPrefix = prefix + available.charAt(i);
                            stack.push(newPrefix);
                            activeTasks.incrementAndGet();
                        }
                    }

                    activeTasks.decrementAndGet();
                }
            } catch (InterruptedException ignored) {}
        };

        // Start workers
        for (int i = 0; i < threadCount; i++) {
            executor.submit(worker);
        }

        // Wait for result or exhaustion
        while (result.get() == null && (activeTasks.get() > 0 || !stack.isEmpty())) {
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                break;
            }
        }

        executor.shutdownNow();
        SwingUtilities.invokeLater(() -> {
            progressBar.setValue((int) currentProgress.get());
            progressBar.setString(currentProgress.get() + "/" + totalCombinations);
        });
        return result.get();
    }

    /**
     * Method for generating all possible MD5 hashes within the given restrictions
     * @param hash User given input
     * @param available String of available characters
     * @param maxLength Length of password
     * @param progressBar Component to update
     * @param totalCombinations Number of total combinations (purely for progress bar functionality
     * @return either cracked password or null if not found
     */
    private static String findMatchingPermutationMD(String hash, String available, int maxLength, JProgressBar progressBar, long totalCombinations) {
        AtomicLong currentProgress = new AtomicLong(0);
        final boolean[] stopRequested = {false};
        AtomicReference<String> result = new AtomicReference<>(null);
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        AtomicLong lastUIUpdateTime = new AtomicLong(System.currentTimeMillis());
        final int UI_UPDATE_STEP = 1000; // update every 1000 steps or 100ms

        // BlockingQueue<String> queue = new LinkedBlockingQueue<>();
        LinkedBlockingDeque<String> stack = new LinkedBlockingDeque<>();
        stack.push(""); // Start with empty prefix

        AtomicInteger activeTasks = new AtomicInteger(1); // Start with 1 for the initial ""
        int threadCount = Runtime.getRuntime().availableProcessors();

        Runnable worker = () -> {
            try {
                while (!stopRequested[0]) {
                    if (result.get() != null && activeTasks.get() == 0) return;

                    // String prefix = queue.poll(100, TimeUnit.MILLISECONDS);
                    String prefix = stack.take();
                    // if (prefix == null) {
                    //     if (activeTasks.get() == 0) return;
                    //     continue;
                    // }

                    if (prefix.length() == maxLength) {
                        // System.out.println(prefix);
                        String computedHash = computeMD5Hash(prefix);
                        long progress = currentProgress.incrementAndGet();
                        long now = System.currentTimeMillis();

                        if (progress % UI_UPDATE_STEP == 0 || (now - lastUIUpdateTime.get()) > 100) {
                            lastUIUpdateTime.set(now);
                            SwingUtilities.invokeLater(() -> {
                                progressBar.setValue((int) progress);
                                progressBar.setString(progress + "/" + totalCombinations);
                            });
                        }


                        if (computedHash.equalsIgnoreCase(hash)) {
                            result.compareAndSet(null, prefix);
                            stopRequested[0] = true;
                        }

                    } else {
                        for (int i = 0; i < available.length(); i++) {
                            String newPrefix = prefix + available.charAt(i);
                            stack.push(newPrefix);
                            activeTasks.incrementAndGet();
                        }
                    }

                    activeTasks.decrementAndGet();
                }
            } catch (InterruptedException ignored) {}
        };

        // Start workers
        for (int i = 0; i < threadCount; i++) {
            executor.submit(worker);
        }

        // Wait for result or exhaustion
        while (result.get() == null && (activeTasks.get() > 0 || !stack.isEmpty())) {
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                break;
            }
        }

        executor.shutdownNow();
        SwingUtilities.invokeLater(() -> {
            progressBar.setValue((int) currentProgress.get());
            progressBar.setString(currentProgress.get() + "/" + totalCombinations);
        });
        return result.get();
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

    private static final ThreadLocal<MessageDigest> sha256DigestThreadLocal = ThreadLocal.withInitial(() -> {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    });

    /**
     * Method to generate the SHA-256 hash
     * @param input String we want to hash
     * @return String representation of a hash
     */
    private static String computeSHA256Hash(String input) {
        MessageDigest md = sha256DigestThreadLocal.get();
        md.reset();
        byte[] hashBytes = md.digest(input.getBytes(StandardCharsets.UTF_8));
        return byteArrayToHexString(hashBytes);
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

    private static final ThreadLocal<MessageDigest> md5DigestThreadLocal = ThreadLocal.withInitial(() -> {
        try {
            return MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    });

    /**
     * Method to generate the MD5 hash
     * @param input String we want to hash
     * @return String representation of a hash
     */
    private static String computeMD5Hash(String input) {
        MessageDigest md = md5DigestThreadLocal.get();
        md.reset(); // Important when reusing!
        byte[] hashBytes = md.digest(input.getBytes(StandardCharsets.UTF_8));
        return byteArrayToHexString(hashBytes);
    }

    /**
     * Method that generates a character set of available character as specified by the user
     * @param opt Integer representation of options
     * @return A String with all possible characters
     */
    public static String getCharacterSet(int opt) {
        switch (opt) {
            case 1 : return MultithreadedSolution.smallAlpha;
            case 2 : return MultithreadedSolution.bigAlpha;
            case 3 : return MultithreadedSolution.smallAlpha + MultithreadedSolution.bigAlpha;
            case 4 : return MultithreadedSolution.nonAlpha;
            case 5 : return MultithreadedSolution.smallAlpha + MultithreadedSolution.nonAlpha;
            case 6 : return MultithreadedSolution.bigAlpha + MultithreadedSolution.nonAlpha;
            default : return MultithreadedSolution.smallAlpha + MultithreadedSolution.bigAlpha + MultithreadedSolution.nonAlpha;
        }
    }
}
