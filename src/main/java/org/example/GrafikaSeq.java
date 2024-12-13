package org.example;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class GrafikaSeq {
    public static void createAndShowGUI() {
        // Create the main frame
        JFrame frame = new JFrame("GrafikaSeq GUI");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(400, 400);

        // Create a panel to hold all components
        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(6, 1)); // 6 rows for the components

        // Add a text input field
        JTextField textField = new JTextField(20);
        panel.add(new JLabel("Enter text:"));
        panel.add(textField);

        // Add radio buttons
        JRadioButton radioButton1 = new JRadioButton("MD-5");
        JRadioButton radioButton2 = new JRadioButton("SHA-256");
        ButtonGroup radioGroup = new ButtonGroup(); // Group the radio buttons
        radioGroup.add(radioButton1);
        radioGroup.add(radioButton2);

        JPanel radioPanel = new JPanel();
        radioPanel.add(radioButton1);
        radioPanel.add(radioButton2);
        panel.add(new JLabel("Select an option:"));
        panel.add(radioPanel);

        // Add checkboxes
        JCheckBox checkbox1 = new JCheckBox("a-z");
        JCheckBox checkbox2 = new JCheckBox("A-Z");
        JCheckBox checkbox3 = new JCheckBox("special");

        JPanel checkboxPanel = new JPanel();
        checkboxPanel.add(checkbox1);
        checkboxPanel.add(checkbox2);
        checkboxPanel.add(checkbox3);
        panel.add(new JLabel("Select options:"));
        panel.add(checkboxPanel);

        // Add an integer menu
        JComboBox<Integer> integerMenu = new JComboBox<>(new Integer[]{1, 2, 3, 4, 5});
        panel.add(new JLabel("Length:"));
        panel.add(integerMenu);

        // Add a button
        JButton button = new JButton("Submit");
        button.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // Get text from the text field
                String text = textField.getText();

                // Get selected radio button
                String selectedRadio = "None";
                if (radioButton1.isSelected()) {
                    selectedRadio = "Option 1";
                } else if (radioButton2.isSelected()) {
                    selectedRadio = "Option 2";
                }

                // Get selected checkboxes
                StringBuilder selectedCheckboxes = new StringBuilder();
                if (checkbox1.isSelected()) {
                    selectedCheckboxes.append("Checkbox 1 ");
                }
                if (checkbox2.isSelected()) {
                    selectedCheckboxes.append("Checkbox 2 ");
                }
                if (checkbox3.isSelected()) {
                    selectedCheckboxes.append("Checkbox 3 ");
                }

                // Get selected integer from the combo box
                Integer selectedInteger = (Integer) integerMenu.getSelectedItem();

                // Print the collected data to standard output
                System.out.println("Text: " + text);
                System.out.println("Selected Radio Button: " + selectedRadio);
                System.out.println("Selected Checkboxes: " + selectedCheckboxes.toString().trim());
                System.out.println("Selected Integer: " + selectedInteger);
            }
        });
        panel.add(button);

        // Add the panel to the frame
        frame.add(panel);

        // Set the frame to be visible
        frame.setVisible(true);
    }
}
